#SEED = 42

import argparse
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import tensorflow as tf
from time import time
from tqdm import tqdm


from modules import ModelDataSet
from modules import ProfileFindingSetup
from modules import SequenceRepresentation
from modules import training
from modules.utils import full_stack


def main():
    parser = argparse.ArgumentParser(description='Run model vs STREME on exon data')
    parser.add_argument('--fasta', help = 'Path to the fasta file containing the sequences', required = True, type = str)
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
    dataset_args.add_argument('--tile-size', help = 'Tile size for the model', required = False, type = int, 
                              default = 334)
    dataset_args.add_argument('--tiles-per-X', help = 'Number of tiles per X', required = False, type = int, 
                              default=7)
    dataset_args.add_argument('--batch-size', help = 'Number of tiles per X', required = False, type = int,
                              default=1)
    dataset_args.add_argument('--prefetch', help = 'Number of batches to prefetch', required = False, type = int,
                              default=3)
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
        assert args.maxseqs > 0, f"[ERROR] >>> Maximum number of exons must be positive, not {args.maxseqs}"
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
    
    genomes = [SequenceRepresentation.Genome([s]) for s in sequences]
    
    # === TRAINING ===

    logging.info("[main] Starting training and evaluation")

    # dump settings to file
    with open(outdir / "settings.json", 'wt') as fh:
        json.dump(settings, fh, indent=2)

    evaluator = training.MultiTrainingEvaluation()
    starttime = time()
    logging.info(f"[main] Prepare training")

    # store single-exon genomes for later evaluation
    with open(os.path.join(outdir, f"training_genomes.json"), 'wt') as fh:
        json.dump([g.toList() for g in genomes], fh)

    # --- train our model ---
    data = ModelDataSet.ModelDataSet(genomes, datamode,
                                     tile_size=args.tile_size, tiles_per_X=args.tiles_per_X,
                                     batch_size=args.batch_size, prefetch=args.prefetch)
    trainsetup = ProfileFindingSetup.ProfileFindingTrainingSetup(data,
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
    try:
        logging.info(f"[main] Start training and evaluation")
        training.trainAndEvaluate(fasta.name, trainsetup, evaluator, 
                                  outdir,  # type: ignore
                                  do_not_train=args.do_not_train,
                                  rand_seed=SEED) # type: ignore
    except Exception as e:
        logging.error(f"[main] trainAndEvaluate failed, check log for details")
        logging.error(f"[main] Error message: {e}")
        logging.debug(full_stack())

    evaluator.dump(os.path.join(outdir, "evaluator.json"))
    if len(evaluator.trainings) > 0:
        evaluator.trainings[0].toMemeTxt(Path(outdir) / "profiles.meme")

    endtime = time()
    runtime = endtime - starttime
    logging.info(f"[main] Finished training and evaluation. Took {runtime:.2f}s")



if __name__ == "__main__":
    main()