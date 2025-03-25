"""
Run our model and STREME on the more raw tfbs peak data in a more straightforward fashion, i.e. 
with training and test sets, maybe k*l-CV, simpler evaluation of Sn, Sp, etc.
"""

#SEED = 42

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import re
import sys
import tensorflow as tf
from time import time
from tqdm import tqdm


from modules import model
from modules import ModelDataSet
from modules import ProfileFindingSetup
from modules import SequenceRepresentation
from modules import Streme
from modules import training
from modules.utils import full_stack


def main():
    parser = argparse.ArgumentParser(description='Run model vs STREME on raw tfbs benchmark data')
    parser.add_argument('--fasta', help = 'Path to the fasta file containing the sequences', required = True, 
                        type = str)
    parser.add_argument('--peaks', help = 'Path to one ore more tsv files with peak information, where the first ' \
                                          + 'column contains the sequence IDs (matching fasta) and the second column ' \
                                          + 'contains a peak position (0-based, relative to fasta sequence start)',
                        required = False, type = str, nargs = '+')
    parser.add_argument('--out', help = 'Output directory', required = True, type = str)
    parser.add_argument('--mode', help="Data mode, either `DNA` or `Translated`", required = True, type = str, 
                        choices = ['DNA', 'Translated'])
    parser.add_argument('--config', help = 'Path to JSON object with training configuration. Allowed keys are all ' \
                        + 'arguments in parsed form, i.e. no leading dashes and inner dashes (-) must be replaced by ' \
                        + 'underscores (_) (e.g. `tile_size` instead of `--tile-size`). Given command line arguments ' \
                        + 'overwrite the values in the config file. For arguments neither supplied via command line ' \
                        + 'call nor the config file, the default values are used.', required = False, type = str)
    parser.add_argument('--maxseqs', help = 'Maximum number of sequences from the to use from the input fasta', 
                        required = False, type = int)
    parser.add_argument('--no-softmasking', help = 'Removes softmasking from sequences before training', 
                        required = False, action = 'store_true')
    parser.add_argument('--do-not-train', help = 'Do not train the model, only evaluate the profiles', required = False,
                        action = 'store_true')
    parser.add_argument('--rand-seed', help = 'Random seed for reproducibility', required = False, type = int)
    # add arguments for model dataset and setup options
    dataset_args = parser.add_argument_group('Dataset options')
    dataset_args.add_argument('--k-cv', help = 'Number of folds for k-fold cross-validation. Set to 1 or lower for no '\
                                             + 'cross-validation', required = False, type = int, default = 0)
    dataset_args.add_argument('--train-fraction', help = 'Fraction of sequences to use for training. Ignored if ' \
                                                         + '--k-cv is set to >= 2', required = False, type = float, 
                              default = 0.7)
    dataset_args.add_argument('--tile-size', help = 'Tile size for the model', required = False, type = int, 
                              default = 334)
    dataset_args.add_argument('--tiles-per-X', help = 'Number of tiles per X', required = False, type = int, 
                              default=7)
    dataset_args.add_argument('--batch-size', help = 'Number of tiles per X', required = False, type = int,
                              default=1)
    dataset_args.add_argument('--prefetch', help = 'Number of batches to prefetch', required = False, type = int,
                              default=3)
    dataset_args.add_argument('--generate-negative-samples', help = 'Generate negative samples for test data',
                              required = False, action = 'store_true')
    # add arguments for model options
    model_args = parser.add_argument_group('Model options')
    model_args.add_argument('--n-best-profiles', help = 'Number of best profiles to report', required = False, 
                            type = int, default = 2)
    model_args.add_argument('--U', help = 'Number of profiles', required = False, type = int, default = 200)
    model_args.add_argument('--enforceU', help = 'Enforce U in profile initialization', required = False, 
                            action = 'store_true')
    model_args.add_argument('--minU', help = 'Only if enforceU is False. Minimum number of profiles to initialize, ' \
                            + 'starting with the most frequent kmers. At most U profiles are initialized.',
                            required = False, type = int, default = 10)
    model_args.add_argument('--minOcc', help = 'Only if enforceU is False. Minimum number of occurences of a kmer to ' \
                            + 'be considered. Is ignored if minU would not be reached otherwise.', required = False, 
                            type = int, default = 8)
    model_args.add_argument('--overlapTilesize', help = 'Maximum overlap of kmers to be ignored in profile ' \
                            + 'initialization', required = False, type = int, default = 6)
    model_args.add_argument('--k', help = 'Length of profiles', required = False, type = int, default = 20)
    model_args.add_argument('--midK', help = 'Length of k-mers to initialize the middle part of profiles', 
                            required = False, type = int, default = 12)
    model_args.add_argument('--s', help = 'Profile shift to both sides', required = False, type = int, default = 0)
    model_args.add_argument('--gamma', help = 'Softmax scale in loss function', required = False, type = float,
                            default = 1.0)
    model_args.add_argument('--l2', help = 'L2 regularization factor in loss function', required = False, type = float,
                            default = 0.01)
    model_args.add_argument('--match-score-factor', help = 'Sites must match a profile at least this fraction of the ' \
                            + 'best matching site to be considered a match', required = False, type = float,
                            default = 0.7)
    model_args.add_argument('--learning-rate', help = 'Learning rate', required = False, type = float, default = 2.0)
    model_args.add_argument('--lr-patience', help = 'Number of epochs to wait for loss decrease before trigger ' \
                            + 'learning rate reduction', required = False, type = int, default = 5)
    model_args.add_argument('--lr-factor', help = 'Factor to reduce learning rate by', required = False, type = float,
                            default = 0.75)
    model_args.add_argument('--rho', help = 'Influence of initial sampling position on profile initialization via ' \
                            + 'seeds', required = False, type = float, default = 0.0)
    model_args.add_argument('--sigma', help = 'Stddev of random normal values added to profile initialization via ' \
                            + 'seeds (mean 0)', required = False, type = float, default = 1.0)
    model_args.add_argument('--phylo-t', help = 'Use prior knowledge on amino acid similarity. Values in [0, 250] ' \
                            + 'are reasonable (0.0 means no prior knowledge). Time a CTMC evolves from the parameter ' \
                            + 'profile P to the profile that is used for scoring/searching. If t==0.0 this prior ' \
                            + 'knowledge is not used. Requires amino acid alphabet, in particular k=20',
                            required = False, type = float, default = 0.0)
    model_args.add_argument('--profile-plateau', help = 'number of epochs to wait for loss plateau to trigger ' \
                            + 'profile reporting', required = False, type = int, default = 10)
    model_args.add_argument('--profile-plateau-dev', help = 'Upper threshold for stddev of loss plateau to trigger ' \
                            + 'profile reporting', required = False, type = float, default = 150)
    args = parser.parse_args()

    # handle arguments

    # https://docs.python.org/3/library/argparse.html#dest --> dest is automatically set to the (first) long option name
    #                                                          and dashes (-) are replaced by underscores (_)

    outdir = Path(args.out) # required, thus always set and cannot be changed by config file
    os.makedirs(outdir, exist_ok=True) # make sure that outdir exists

    # set logfile
    logging.basicConfig(filename = outdir / "logfile.txt",
                        format="%(asctime)s %(levelname)s: %(message)s", 
                        encoding='utf-8', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("TensorFlow version: "+str(tf.__version__))

    # load config file if given and overwrite default values
    if args.config is not None: # drawback: no type checking on config file values
        conffile = Path(args.config)
        assert conffile.is_file(), f"[ERROR] >>> Config file {conffile} not found"
        with open(conffile, 'rt') as fh:
            config = json.load(fh)

        # overwrite default values with config values, keeping command line values if given
        for key in config:
            if key in vars(args):
                arg = "--"+key.replace("_", "-")
                if arg in sys.argv:
                    logging.warning(f"[main] Argument '{arg}' is set via command line and config file, using command " \
                                    + f"line value {vars(args)[key]}")
                else:
                    logging.info(f"[main] Set argument '{arg}' to value {config[key]} from config file")
                    setattr(args, key, config[key])
            else:
                logging.warning(f"[main] Unknown key '{key}' in config file, ignoring")

    # store arguments in a settings dict for later reference
    settings = vars(args)

    # handle arguments
    if args.rand_seed is not None:
        SEED = args.rand_seed
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(SEED)
    else:
        SEED = None

    fasta = Path(args.fasta)
    assert fasta.is_file(), f"[ERROR] >>> Input file '{fasta}' not found"
    if args.maxseqs is not None:
        assert args.maxseqs > 0, f"[ERROR] >>> Maximum number of sequences must be positive, not {args.maxseqs}"
        MAXSEQS = args.maxseqs
    else:
        MAXSEQS = None

    if args.mode == 'DNA':
        datamode = ModelDataSet.DataMode.DNA
    else:
        if args.phylo_t == 0:
            datamode = ModelDataSet.DataMode.Translated
        else:
            logging.warning("[main] Phylo_t is not 0.0 and data mode is set to 'Translated'. Setting data mode to " \
                            + "'Translated_noStop', using only 20-letter aa alphabet without stop codon.")
            datamode = ModelDataSet.DataMode.Translated_noStop
            
    logging.info(f"[main] Data mode: {datamode}")

    # === LOAD DATA ===

    logging.info("[main] Loading sequences")
    sequences = SequenceRepresentation.loadFasta_agnostic(fasta)
    if MAXSEQS is not None and MAXSEQS < len(sequences):
        logging.info(f"[main] Limiting data to {MAXSEQS}/{len(sequences)} sequences from the input fasta")
        sequences = sequences[:MAXSEQS]

    # load peak data if given
    peaks = None
    if args.peaks is not None:
        logging.debug(f"[main] Loading peaks from {args.peaks}")
        # loadFasta_agnostic stores the fasta id as species attribute in the Sequence objects, but the IDs of the
        # Sequence objects probably differ. Thus create a mapping here to be able to match the peak data to the
        # sequences
        faid2seqid = {s.species: s.id for s in sequences}
        for peakfile in args.peaks:
            try:
                assert Path(peakfile).is_file(), f"[ERROR] >>> Peak file '{peakfile}' not found"
                df = pd.read_csv(peakfile, sep='\t', header=None, names=['seqid', 'peakpos'])
                assert df.shape[1] >= 2, f"[ERROR] >>> Peak file '{peakfile}' must have at least two columns"
                # only keep sequences that are in the fasta file
                #logging.debug(f"[main] {len(df.index)} peaks from {peakfile}")
                df = df[df['seqid'].isin(faid2seqid.keys())]
                #logging.debug(f"[main] {len(df.index)} remaining peaks after filtering from {peakfile}")
                df['seqid'] = df['seqid'].map(faid2seqid) # replace the fasta IDs with the sequence IDs
                if peaks is None:
                    df['source'] = Path(peakfile).name
                    peaks = df
                else:
                    src = Path(peakfile).name
                    if src in peaks['source'].values:
                        logging.warning(f"[main] Peak file with same name as '{peakfile}' has already been loaded, " \
                                        + "using full path as source name instead")
                        src = peakfile
                    df['source'] = src
                    peaks = pd.concat([peaks, df], ignore_index=True)

            except Exception as e:
                logging.error(f"[main] Error while processing peak file '{peakfile}', check log for details")
                logging.error(f"[main] Error message: {e}")
                logging.debug(full_stack())

    if peaks is not None:
        # add the peaks as genomic elements to the sequences, then they will get drawn as well via geneLinkDraw
        logging.debug(f"[main] Adding peaks (total: {len(peaks.index)}) to sequences")
        for seq in sequences:
            peakdf = peaks[peaks['seqid'] == seq.id]
            #logging.debug(f"[main] {len(peakdf.index)} peaks to {seq.id}")
            for peaksrc in peakdf['source'].unique():
                for peak in peakdf[peakdf['source'] == peaksrc]['peakpos'].values:
                    if peak < 0 or peak >= seq.length:
                        logging.warning(f"[main] Peak position {peak} is out of bounds for sequence {seq.id}, " \
                                        + f"skipping")
                        continue
                    
                    seq.addSubsequenceAsElement(start=int(peak), end=int(peak)+1, seqtype=f"peak_{peaksrc}",
                                                source=str(peaksrc), genomic_positions=False)
    
    # continue creating training data
    genomes = [SequenceRepresentation.Genome([s]) for s in sequences]

    # make splits (either k-fold or train/test split)
    genomes = random.sample(genomes, len(genomes)) # shuffle genomes
    data_splits: list[dict[str, list[SequenceRepresentation.Genome]]] = []
    if args.k_cv > 1:
        assert args.k_cv < len(genomes), f"[ERROR] >>> Number of folds must be less than number of sequences, " \
                                         + f"not {args.k_cv} >= {len(genomes)}"
        fold_size = len(genomes) // args.k_cv
        for i in range(args.k_cv):
            test_start = i * fold_size
            test_end = (i+1) * fold_size
            data_splits.append({'train': genomes[:test_start] + genomes[test_end:], 
                                'test': genomes[test_start:test_end]})
    else:
        assert args.train_fraction > 0 and args.train_fraction <= 1, f"[ERROR] >>> Train fraction must be in (0, 1], " \
                                                                     + f"not {args.train_fraction}"
        train_size = int(args.train_fraction * len(genomes))
        data_splits.append({'train': genomes[:train_size], 'test': genomes[train_size:]})

    if args.generate_negative_samples:
        logging.info("[main] Generating negative samples for test data")
        def _generate_negative_sequence(len, nt_bg_dist):
            seq = ''
            for i in range(len):
                seq += np.random.choice(list(nt_bg_dist.keys()), p=list(nt_bg_dist.values()))
            return seq

        for i in range(len(data_splits)):
            split = data_splits[i]
            testgenomes: list[SequenceRepresentation.Genome] = split['test']
            nt_bg_dist = {nt: 0 for nt in 'ACGT'}
            for g in testgenomes:
                for s in g:
                    for nt in 'ACGT':
                        nt_bg_dist[nt] += s.sequence.count(nt)
            nt_bg_dist = {nt: nt_bg_dist[nt] / sum(nt_bg_dist.values()) for nt in nt_bg_dist}
            neg_genomes = []
            for g in testgenomes:
                neg_seqs = []
                for s in g:
                    neg_seqs.append(SequenceRepresentation.Sequence(species=f"negative_{s.species}", 
                                                                    chromosome=s.chromosome, 
                                                                    strand=s.strand, 
                                                                    genome_start=0,
                                                                    sequence=_generate_negative_sequence(s.length, 
                                                                                                         nt_bg_dist)))
                neg_genomes.append(SequenceRepresentation.Genome(neg_seqs))

            data_splits[i]['test_negative'] = neg_genomes
                
    
    # === TRAINING ===

    logging.info("[main] Starting training and evaluation")

    # dump settings to file
    with open(outdir / "settings.json", 'wt') as fh:
        json.dump(settings, fh, indent=2)

    # # dump peaks to file if given (not needed for now, peaks are stored in the sequences)
    # if peaks is not None:
    #     peaks.to_csv(outdir / "peaks.tsv", sep='\t', index=False)

    model_evaluator_train = training.MultiTrainingEvaluation()
    model_evaluator_test = training.MultiTrainingEvaluation()
    model_evaluator_negtest = training.MultiTrainingEvaluation() # for negative test data
    streme_evaluator = training.MultiTrainingEvaluation()
    streme_evaluator_train = training.MultiTrainingEvaluation() # for dummy-model evaluation
    streme_evaluator_test = training.MultiTrainingEvaluation()
    streme_evaluator_negtest = training.MultiTrainingEvaluation() # for negative test data dummy-model evaluation
    starttime = time()

    for splitidx, split in enumerate(data_splits):
        logging.info(f"[main] Prepare training {splitidx+1}/{len(data_splits)}")

        train_sequences = split['train']
        test_sequences = split['test']
        negative_test_sequences = split.get('test_negative', None)
        # dump test and negative sequences to fasta for fimo/mast site search after training
        SequenceRepresentation.sequenceListToFASTA([s for g in test_sequences for s in g],
                                                   str(outdir / f"test_sequences_{splitidx}.fasta"))
        if negative_test_sequences is not None:
            SequenceRepresentation.sequenceListToFASTA([s for g in negative_test_sequences for s in g], 
                                                       str(outdir / f"negative_test_sequences_{splitidx}.fasta"))
        # store sequences for later evaluation
        with open(os.path.join(outdir, f"training_sequences_{splitidx}.json"), 'wt') as fh:
            json.dump([g.toList() for g in train_sequences], fh)
        with open(os.path.join(outdir, f"test_sequences_{splitidx}.json"), 'wt') as fh:
            json.dump([g.toList() for g in test_sequences], fh)
        if negative_test_sequences is not None:
            with open(os.path.join(outdir, f"negative_test_sequences_{splitidx}.json"), 'wt') as fh:
                json.dump([g.toList() for g in negative_test_sequences], fh)

        traindata = ModelDataSet.ModelDataSet(train_sequences, datamode,
                                              tile_size=args.tile_size, tiles_per_X=args.tiles_per_X,
                                              batch_size=args.batch_size, prefetch=args.prefetch)
        
        # --- train our model ---
        trainsetup = ProfileFindingSetup.ProfileFindingTrainingSetup(traindata,
                                                                     U = args.U, k = args.k, 
                                                                     midK = args.midK, s = args.s, 
                                                                     epochs = 350, gamma = args.gamma, l2 = args.l2,
                                                                     match_score_factor = args.match_score_factor,
                                                                     learning_rate = args.learning_rate,
                                                                     lr_patience = args.lr_patience,
                                                                     lr_factor = args.lr_factor,
                                                                     rho = args.rho, sigma = args.sigma,
                                                                     profile_plateau = args.profile_plateau,
                                                                     profile_plateau_dev = args.profile_plateau_dev,
                                                                     n_best_profiles = args.n_best_profiles,
                                                                     phylo_t = args.phylo_t)
        trainsetup.initializeProfiles_kmers(enforceU=args.enforceU, 
                                            minU=args.minU, minOcc=args.minOcc,
                                            overlapTilesize=args.overlapTilesize,
                                            plot=False)

        testdata = ModelDataSet.ModelDataSet(test_sequences, datamode,
                                             tile_size=args.tile_size, tiles_per_X=args.tiles_per_X,
                                             batch_size=args.batch_size, prefetch=args.prefetch)
        if negative_test_sequences is not None:
            negtestdata = ModelDataSet.ModelDataSet(negative_test_sequences, datamode,
                                                    tile_size=args.tile_size, tiles_per_X=args.tiles_per_X,
                                                    batch_size=args.batch_size, prefetch=args.prefetch)
        else:
            negtestdata = None

        try:
            logging.info(f"[main] Start model training and evaluation {splitidx+1}/{len(data_splits)}")
            training.trainAndTest(fasta.name, trainsetup, testdata, 
                                  model_evaluator_train, model_evaluator_test, 
                                  outdir,  # type: ignore
                                  do_not_train=args.do_not_train,
                                  rand_seed=SEED,
                                  linkplot_single_genecol="lightgray",
                                  linkplot_single_linkcol="indigo",
                                  linkplot_genewidth=20,
                                  linkplot_linkwidth=10,
                                  sitecols={
                                      'kmer sites': "#00ff007f",
                                      'masking sites': "#00ff001a",
                                      },
                                  elementcols={
                                      'peak_bed.tsv': "darkred",
                                      'peak_fimo.tsv': "darkorange",
                                      'peak_mast.tsv': "red",
                                      })
        except Exception as e:
            logging.error(f"[main] trainAndEvaluate failed, check log for details")
            logging.error(f"[main] Error message: {e}")
            logging.debug(full_stack())

        model_evaluator_train.dump(str(outdir / "evaluator_training.json"))
        model_evaluator_test.dump(str(outdir / "evaluator_test.json"))
        if len(model_evaluator_train.trainings) > splitidx:
            model_evaluator_train.trainings[splitidx].toMemeTxt(outdir / f"profiles_{splitidx}.meme")

        # evaluate model with negative data, using dummy model
        if negative_test_sequences is not None and len(model_evaluator_train.trainings) > splitidx:
            motifwrapper = model_evaluator_train.trainings[splitidx].motifs
            training.testMotifs(fasta.name, motifwrapper,
                                traindata, negtestdata, args.match_score_factor,
                                training.MultiTrainingEvaluation(), model_evaluator_negtest,
                                outdir=str(outdir), outprefix=f"model_{splitidx}_dummymodel_negative_",
                                linkplot_single_genecol="lightgray",
                                linkplot_single_linkcol="indigo",
                                linkplot_genewidth=20,
                                linkplot_linkwidth=10,
                                sitecols={
                                    'kmer sites': "#00ff007f",
                                    'masking sites': "#00ff001a",
                                    },
                                elementcols={
                                    'peak_bed.tsv': "darkred",
                                    'peak_fimo.tsv': "darkorange",
                                    'peak_mast.tsv': "red",
                                    })
            model_evaluator_negtest.dump(str(outdir / "evaluator_negative_test.json"))

        # --- train STREME ---
        streme = Streme.Streme(working_dir=str(outdir / "STREME"),
                               k_min=args.k, k_max=args.k, 
                               n_best_motifs=args.n_best_profiles)
        try:
            logging.info(f"[main] Start STREME training and evaluation {splitidx+1}/{len(data_splits)}")
            streme.run(splitidx, traindata, streme_evaluator, plot_motifs=True, plot_links=True, verbose=True)
        except Exception as e:
            logging.error(f"[main] STREME failed, check log for details")
            logging.error(f"[main] Error message: {e}")
            logging.debug(full_stack())
        
        streme_evaluator.dump(str(outdir / "STREME" / "streme_evaluator.json"))

        # evaluate STREME motifs with dummy model
        if len(streme_evaluator.trainings) > splitidx:
            motifwrapper = streme_evaluator.trainings[splitidx].motifs
            training.testMotifs(fasta.name, motifwrapper,
                                traindata, testdata, args.match_score_factor,
                                streme_evaluator_train, streme_evaluator_test,
                                outdir=str(outdir / "STREME"), outprefix=f"streme_{splitidx}_dummymodel_",
                                linkplot_single_genecol="lightgray",
                                linkplot_single_linkcol="indigo",
                                linkplot_genewidth=20,
                                linkplot_linkwidth=10,
                                sitecols={
                                    'kmer sites': "#00ff007f",
                                    'masking sites': "#00ff001a",
                                    },
                                elementcols={
                                    'peak_bed.tsv': "darkred",
                                    'peak_fimo.tsv': "darkorange",
                                    'peak_mast.tsv': "red",
                                    })
            streme_evaluator_train.dump(str(outdir / "STREME" / "streme_evaluator_dummymodel_train.json"))
            streme_evaluator_test.dump(str(outdir / "STREME" / "streme_evaluator_dummymodel_test.json"))

            if negative_test_sequences is not None:
                training.testMotifs(fasta.name, motifwrapper,
                                    traindata, negtestdata, args.match_score_factor,
                                    training.MultiTrainingEvaluation(), streme_evaluator_negtest,
                                    outdir=str(outdir / "STREME"), outprefix=f"streme_{splitidx}_dummymodel_negative_",
                                    linkplot_single_genecol="lightgray",
                                    linkplot_single_linkcol="indigo",
                                    linkplot_genewidth=20,
                                    linkplot_linkwidth=10,
                                    sitecols={
                                        'kmer sites': "#00ff007f",
                                        'masking sites': "#00ff001a",
                                        },
                                    elementcols={
                                        'peak_bed.tsv': "darkred",
                                        'peak_fimo.tsv': "darkorange",
                                        'peak_mast.tsv': "red",
                                        })
                streme_evaluator_negtest.dump(str(outdir / "STREME" / "streme_evaluator_dummymodel_negative_test.json"))
            

    endtime = time()
    runtime = endtime - starttime
    logging.info(f"[main] Finished training and evaluation. Took {runtime:.2f}s")



if __name__ == "__main__":
    main()