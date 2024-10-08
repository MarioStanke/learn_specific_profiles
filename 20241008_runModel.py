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
from modules import Streme
from modules import training
from modules.utils import full_stack



# # HOTFIX: make exon annotations unique, i.e. if the exact same region is annotated multiple times, keep only one annotation
# def bestAnnot(annotation):
#     """ `annotation` might be None, a string or a dict. Check if 'BestRefSeq' is part of the source """
#     if annotation is None:
#         return False
#     elif type(annotation) is str:
#         return re.search("bestrefseq", annotation) is not None
#     else:
#         assert type(annotation) is dict, \
#             f"[ERROR] >>> Type of annotation {annotation} is not dict but {type(annotation)}"
#         if "source" in annotation:
#             return re.search("bestrefseq", annotation["source"]) is not None
        
#         return False



# def makeAnnotationsUnique(genomes: list[SequenceRepresentation.Genome]):
#     """ If a sequence in a genome has mutliple identical annotations w.r.t. the annotated positions, remove that 
#         redundancy by trying to choose the best annotation """
#     stats = {'altered_genomes': set(), 'altered_sequences': set(), 'nduplicates': 0, 'nremoved': 0}
#     for genome in genomes:
#         for sequence in genome:
#             assert sequence.elementsPossible, f"[ERROR] >>> No elements possible in sequence '{sequence}'"
#             if len(sequence.genomic_elements) <= 1:
#                 continue
                
#             elements = {} # genome region to annotations
#             for element in sequence.genomic_elements:
#                 # without the "source", so identical annotations from different sources should group
#                 key = element.toTuple()[:-3] 
#                 if key not in elements:
#                     elements[key] = []
                    
#                 elements[key].append(element)
                
#             # reduce redundant annotations
#             for key in elements:
#                 if len(elements[key]) > 1:
#                     stats['altered_genomes'].add(genome.species)
#                     stats['altered_sequences'].add(sequence.id)
#                     stats['nduplicates'] += 1
#                     stats['nremoved'] += len(elements[key])-1
#                     annotQuality = [bestAnnot(element.source) for element in elements[key]]
#                     annotation = elements[key][annotQuality.index(True)] if any(annotQuality) else elements[key][0]
#                     elements[key] = annotation
#                 else:
#                     elements[key] = elements[key][0]
                    
#             assert all([type(elements[key]) is not list for key in elements]), \
#                 f"[ERROR] >>> Not all elements unique: {[type(elements[key]) for key in elements]},\n\n {elements}"
#             sequence.genomic_elements = list(elements.values())
#             assert len(sequence.genomic_elements) == len(elements), "[ERROR] >>> new annotation length and elements " \
#                                                           + f"length differ: {sequence.genomic_elements} vs. {elements}"
            
#     logging.info(f"[makeAnnotationsUnique] Found and uniq-ed {stats['nduplicates']} redundant annotations in " \
#                  +f"{len(stats['altered_sequences'])} sequences from {len(stats['altered_genomes'])} genomes; "\
#                  +f"removed total of {stats['nremoved']} redundant annotations")



# def selectLongestTranscript(genomes: list[SequenceRepresentation.Genome]):
#     """ If a sequence in a genome has mutliple annotations, check if any two of them overlap such that the shorter is 
#         completely inside the longer one. If such a pair is found, discard the shorter annotation """
#     stats = {'altered_genomes': set(), 'altered_sequences': set(), 'noverlaps': 0, 'nremoved': 0}
#     for genome in genomes:
#         for sequence in genome:
#             assert sequence.elementsPossible, f"[ERROR] >>> No elements possible in sequence '{sequence}'"
#             if len(sequence.genomic_elements) <= 1:
#                 continue
                
#             elements = sorted(sequence.genomic_elements, key = lambda s: s.length) # increasing sequence length
#             longest_elements = []
#             for i in range(len(elements)):
#                 has_superseq = False
#                 for j in range(i+1, len(elements)):
#                     if elements[i].isSubsequenceOf(elements[j]) and elements[i].strand == elements[j].strand:
#                         has_superseq = True
#                         break
                    
#                 if not has_superseq:
#                     longest_elements.append(elements[i])
                    
#             assert len(longest_elements) >= 1, "[ERROR] >>> selectLongestTranscript has filtered out all elements"
#             assert len(longest_elements) <= len(elements),"[ERROR] >>> selectLongestTranscript has duplicated something"
#             if len(elements) > len(longest_elements):
#                 stats['nremoved'] += len(elements) - len(longest_elements)
#                 stats['altered_sequences'].add(sequence.id)
#                 stats['altered_genomes'].add(genome.species)
                
            
#             sequence.genomic_elements = longest_elements
            
#     logging.info("[selectLongestTranscript] Found and removed subsequence annotations in " \
#                  + f"{len(stats['altered_sequences'])} sequences from {len(stats['altered_genomes'])} genomes; " \
#                  + f"removed total of {stats['nremoved']} subsequence annotations")



# def checkUniqueAnnotations(genomes: list[SequenceRepresentation.Genome]):
#     """ Basically check if makeAnnotationsUnique() has worked """
#     allUnique = True
#     for genome in genomes:
#         for sequence in genome:
#             assert sequence.elementsPossible, f"[ERROR] >>> No elements possible in sequence '{sequence}'"
#             if len(sequence.genomic_elements) <= 1:
#                 continue
                
#             elements = {} # genome region to annotations
#             for element in sequence.genomic_elements:
#                 # without the "source", so identical annotations from different sources should group
#                 key = element.toTuple()[:-3] 
#                 if key not in elements:
#                     elements[key] = []
                    
#                 elements[key].append(element)
                
#             # reduce redundant annotations
#             for key in elements:
#                 if len(elements[key]) > 1:
#                     allUnique = False

#     return allUnique



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
    # exonsetFiles = []
    # warnings = []
    # logging.info("[main] Getting exon list")
    # for d in os.listdir(datadir):
    #     subdir = datadir / d
    #     if re.match(r"exon_chr.+", d) and subdir.is_dir():
    #         seqDataFile = subdir / "profile_finding_sequence_data.json"
    #         if not seqDataFile.is_file():
    #             warnings.append(f"[main] Expected file {seqDataFile} but not found, skipping")
    #             continue
                
    #         exonsetFiles.append(seqDataFile)

    # logging.info(f"[main] {len(warnings)} warnings")
    # for warning in warnings:
    #     logging.warning(warning)
        
    # logging.info(f"[main] Number of exons: {len(exonsetFiles)}")
    
    # # shuffle exon list
    # random.shuffle(exonsetFiles)
    
    if MAXSEQS is not None and MAXSEQS < len(sequences):
        logging.info(f"[main] Limiting data to {MAXSEQS}/{len(sequences)} sequences from the input fasta")
        sequences = sequences[:MAXSEQS]
    
    genomes = [SequenceRepresentation.Genome([s]) for s in sequences]
    # singleExonGenomes: list[list[SequenceRepresentation.Genome]] = [] # run training on these
    # stoi = {} # species to allGenomes list index
    # for sl in exonsetFiles:
    #     sequences = SequenceRepresentation.loadJSONSequenceList(str(sl))

    #     # assert unique sequences and unique species (i.e. only one sequence per species)
    #     seqIDs = [seq.id for seq in sequences]
    #     species = [seq.species for seq in sequences]
    #     assert len(seqIDs) == len(set(seqIDs)), f"[ERROR] >>> Duplicate sequences in {sequences}"
    #     assert len(species) == len(set(species)), f"[ERROR] >>> Duplicate species in {sequences}"
    #     assert 'Homo_sapiens' in species, f"[ERROR] >>> Homo sapiens not in {sequences}"
    #     assert species[0] == 'Homo_sapiens', f"[ERROR] >>> Homo sapiens not first in {sequences}"

    #     skip = False
    #     if len(sequences) <= 1:
    #         logging.warning(f"[main] Skipping exon set {[s.id for s in sequences]} as there are not enoug sequences.")
    #         skip = True
    #     else:
    #         for seq in sequences:
    #             assert seq.hasHomologies(), f"[ERROR] >>> Sequence {seq} has no homologies"
    #             assert len(seq.homology) == 1, f"[ERROR] >>> Sequence {seq} has not exactly one homology"

    #             if seq.species == 'Homo_sapiens':
    #                 assert seq.elementsPossible(), f"[ERROR] >>> Sequence {seq} cannot have annotations"
    #                 if len(seq.genomic_elements) == 0:
    #                     logging.warning(f"[main] Skipping sequence {seq} as it has no annotations")
    #                     skip = True

    #                 break

    #     if skip:
    #         continue

    #     # create list of single-exon Genomes for training
    #     seGen = []
    #     for seq in sequences:
    #         # make annotations and transkripts unique
    #         SequenceRepresentation.makeAnnotationsUnique(seq)
    #         SequenceRepresentation.selectLongestTranscript(seq)
    #         if args.no_softmasking:
    #             seq.removeSoftmasking()

    #         seGen.append( SequenceRepresentation.Genome([seq]) )

    #     singleExonGenomes.append(seGen)
        
    #     # store all sequences at their respective genomes
    #     for seq in sequences:
    #         if seq.species not in stoi:
    #             stoi[seq.species] = len(allGenomes) # add new genome, remember index
    #             allGenomes.append(SequenceRepresentation.Genome())

    #         allGenomes[stoi[seq.species]].addSequence(seq)
    
    # logging.info("[main] Number of homologies: "+str(len(exonsetFiles)))
    # logging.info("[main] Total number of genomes: "+str(len(allGenomes)))
    # logging.info("[main] Total number of sequences: "+str(sum([len(g) for g in allGenomes])))
    # logging.info("[main] Sequences per genome: "+\
    #              ', '.join([f"{[len(g) for g in allGenomes].count(i)} genomes with {i} sequences"\
    #                           for i in range(max([len(g) for g in allGenomes])) \
    #                           if [len(g) for g in allGenomes].count(i) > 0]))

    # logging.info("[main] Check unique annotations: "+str(checkUniqueAnnotations(allGenomes)))

    # # store all genomes for later evaluation
    # with open(os.path.join(outdir, "allGenomes.json"), 'wt') as fh:
    #     json.dump([g.toList() for g in allGenomes], fh)

    # === TRAINING ===

    logging.info("[main] Starting training and evaluation")

    # some global settings
    # genome_limit = 50
    # settings['main.genome_limit'] = genome_limit # add to settings dict for later reference
    
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
                                  outdir, rand_seed=SEED) # type: ignore
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

    # training iteration
    # evaluator = training.MultiTrainingEvaluation()
    # streme_evaluator = training.MultiTrainingEvaluation()
    # for i, seGenomes in tqdm(enumerate(singleExonGenomes)):
        # runID = f"{i:04}" # 0000, 0001, ...
        # starttime = time()
        # logging.info(f"[main] Start training and evaluation for run {runID}")

        # seGenomes = seGenomes[:min(len(seGenomes), genome_limit)]
        # # store single-exon genomes for later evaluation
        # with open(os.path.join(outdir, f"{runID}_singleExonGenomes.json"), 'wt') as fh:
        #     json.dump([g.toList() for g in seGenomes], fh)

        # # --- train our model ---
        # logging.info(f"[main] Start training and evaluation on model for {runID}")
        # data = ModelDataSet.ModelDataSet(seGenomes, datamode,
        #                                  tile_size=args.tile_size, tiles_per_X=args.tiles_per_X,
        #                                  batch_size=args.batch_size, prefetch=args.prefetch)
        # trainsetup = ProfileFindingSetup.ProfileFindingTrainingSetup(data,
        #                                                              U = args.U, k = args.k, 
        #                                                              midK = args.midK, s = args.s, 
        #                                                              epochs = 350, gamma = args.gamma, l2 = args.l2,
        #                                                              match_score_factor = args.match_score_factor,
        #                                                              learning_rate = args.learning_rate,
        #                                                              lr_patience = args.lr_patience,
        #                                                              lr_factor = args.lr_factor,
        #                                                              rho = args.rho, sigma = args.sigma,
        #                                                              profile_plateau = args.profile_plateau,
        #                                                              profile_plateau_dev = args.profile_plateau_dev,
        #                                                              n_best_profiles = args.n_best_profiles,
        #                                                              phylo_t = args.phylo_t)
        # trainsetup.initializeProfiles_kmers(enforceU=args.enforceU, 
        #                                     minU=args.minU, minOcc=args.minOcc,
        #                                     overlapTilesize=args.overlapTilesize,
        #                                     plot=False)
        # try:
        #     training.trainAndEvaluate(runID, trainsetup, evaluator, 
        #                               outdir, outprefix=f"{runID}_",  # type: ignore
        #                               rand_seed=SEED) # type: ignore
        # except Exception as e:
        #     logging.error(f"[main] trainAndEvaluate failed for homology {i}, check log for details")
        #     logging.error(f"[main] Error message: {e}")
        #     logging.debug(full_stack())
        #     continue

        # evaluator.dump(os.path.join(outdir, "evaluator.json")) # save after each run

        # # --- run STREME ---
        # logging.info(f"[main] Start training and evaluation on STREME for {runID}")
        # streme_wd = os.path.join(outdir, f"{runID}_STREME")
        # streme_runner = Streme.Streme(working_dir = streme_wd,
        #                               k_min = args.k, k_max = args.k,
        #                               n_best_motifs = args.n_best_profiles,
        #                               load_streme_script= "/home/ebelm/Software/load_MEME.sh")
        # data = ModelDataSet.ModelDataSet(seGenomes, datamode,
        #                                  tile_size=args.tile_size, tiles_per_X=args.tiles_per_X,
        #                                  batch_size=args.batch_size, prefetch=args.prefetch,
        #                                  replaceSpaceWithX=True) # reset data
        # try:
        #     _ = streme_runner.run(runID, data, streme_evaluator, verbose=True, 
        #                           plot_links=True, plot_onlyLinkedSeqs=False)
        # except Exception as e:
        #     logging.error(f"[main] STREME failed for homology {i}, check log for details")
        #     logging.error(f"[main] Error message: {e}")
        #     logging.debug(full_stack())
        #     continue
        
        # streme_evaluator.dump(os.path.join(outdir, "streme_evaluator.json")) # save after each run

        # endtime = time()
        # runtime = endtime - starttime
        # logging.info(f"[main] Finished training and evaluation for {runID}. Took {runtime:.2f}s")



if __name__ == "__main__":
    main()