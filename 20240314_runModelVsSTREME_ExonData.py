SEED = 42

import argparse
from datetime import datetime
import json
import logging
import os
import random
import re
import tensorflow as tf
from time import time
from tqdm import tqdm
from IPython.display import Audio

if SEED is not None:
    # enable deterministic tensorflow behaviour (https://github.com/NVIDIA/framework-determinism/blob/master/doc/tensorflow.md)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(SEED)

from modules import SequenceRepresentation
from modules import Streme
from modules import ProfileFindingSetup
from modules import training



# HOTFIX: make exon annotations unique, i.e. if the exact same region is annotated multiple times, keep only one annotation
def bestAnnot(annotation):
    """ `annotation` might be None, a string or a dict. Check if 'BestRefSeq' is part of the source """
    if annotation is None:
        return False
    elif type(annotation) is str:
        return re.search("bestrefseq", annotation) is not None
    else:
        assert type(annotation) is dict, \
            f"[ERROR] >>> Type of annotation {annotation} is not dict but {type(annotation)}"
        if "source" in annotation:
            return re.search("bestrefseq", annotation["source"]) is not None
        
        return False



def makeAnnotationsUnique(genomes: list[SequenceRepresentation.Genome]):
    """ If a sequence in a genome has mutliple identical annotations w.r.t. the annotated positions, remove that 
        redundancy by trying to choose the best annotation """
    stats = {'altered_genomes': set(), 'altered_sequences': set(), 'nduplicates': 0, 'nremoved': 0}
    for genome in genomes:
        for sequence in genome:
            assert sequence.elementsPossible, f"[ERROR] >>> No elements possible in sequence '{sequence}'"
            if len(sequence.genomic_elements) <= 1:
                continue
                
            elements = {} # genome region to annotations
            for element in sequence.genomic_elements:
                # without the "source", so identical annotations from different sources should group
                key = element.toTuple()[:-3] 
                if key not in elements:
                    elements[key] = []
                    
                elements[key].append(element)
                
            # reduce redundant annotations
            for key in elements:
                if len(elements[key]) > 1:
                    stats['altered_genomes'].add(genome.species)
                    stats['altered_sequences'].add(sequence.id)
                    stats['nduplicates'] += 1
                    stats['nremoved'] += len(elements[key])-1
                    annotQuality = [bestAnnot(element.source) for element in elements[key]]
                    annotation = elements[key][annotQuality.index(True)] if any(annotQuality) else elements[key][0]
                    elements[key] = annotation
                else:
                    elements[key] = elements[key][0]
                    
            assert all([type(elements[key]) is not list for key in elements]), \
                f"[ERROR] >>> Not all elements unique: {[type(elements[key]) for key in elements]},\n\n {elements}"
            sequence.genomic_elements = list(elements.values())
            assert len(sequence.genomic_elements) == len(elements), "[ERROR] >>> new annotation length and elements " \
                                                          + f"length differ: {sequence.genomic_elements} vs. {elements}"
            
    logging.info(f"[makeAnnotationsUnique] Found and uniq-ed {stats['nduplicates']} redundant annotations in " \
                 +f"{len(stats['altered_sequences'])} sequences from {len(stats['altered_genomes'])} genomes; "\
                 +f"removed total of {stats['nremoved']} redundant annotations")



def selectLongestTranscript(genomes: list[SequenceRepresentation.Genome]):
    """ If a sequence in a genome has mutliple annotations, check if any two of them overlap such that the shorter is 
        completely inside the longer one. If such a pair is found, discard the shorter annotation """
    stats = {'altered_genomes': set(), 'altered_sequences': set(), 'noverlaps': 0, 'nremoved': 0}
    for genome in genomes:
        for sequence in genome:
            assert sequence.elementsPossible, f"[ERROR] >>> No elements possible in sequence '{sequence}'"
            if len(sequence.genomic_elements) <= 1:
                continue
                
            elements = sorted(sequence.genomic_elements, key = lambda s: s.length) # increasing sequence length
            longest_elements = []
            for i in range(len(elements)):
                has_superseq = False
                for j in range(i+1, len(elements)):
                    if elements[i].isSubsequenceOf(elements[j]) and elements[i].strand == elements[j].strand:
                        has_superseq = True
                        break
                    
                if not has_superseq:
                    longest_elements.append(elements[i])
                    
            assert len(longest_elements) >= 1, "[ERROR] >>> selectLongestTranscript has filtered out all elements"
            assert len(longest_elements) <= len(elements),"[ERROR] >>> selectLongestTranscript has duplicated something"
            if len(elements) > len(longest_elements):
                stats['nremoved'] += len(elements) - len(longest_elements)
                stats['altered_sequences'].add(sequence.id)
                stats['altered_genomes'].add(genome.species)
                
            
            sequence.genomic_elements = longest_elements
            
    logging.info("[selectLongestTranscript] Found and removed subsequence annotations in " \
                 + f"{len(stats['altered_sequences'])} sequences from {len(stats['altered_genomes'])} genomes; " \
                 + f"removed total of {stats['nremoved']} subsequence annotations")



def checkUniqueAnnotations(genomes: list[SequenceRepresentation.Genome]):
    """ Basically check if makeAnnotationsUnique() has worked """
    allUnique = True
    for genome in genomes:
        for sequence in genome:
            assert sequence.elementsPossible, f"[ERROR] >>> No elements possible in sequence '{sequence}'"
            if len(sequence.genomic_elements) <= 1:
                continue
                
            elements = {} # genome region to annotations
            for element in sequence.genomic_elements:
                # without the "source", so identical annotations from different sources should group
                key = element.toTuple()[:-3] 
                if key not in elements:
                    elements[key] = []
                    
                elements[key].append(element)
                
            # reduce redundant annotations
            for key in elements:
                if len(elements[key]) > 1:
                    allUnique = False

    return allUnique



def main():
    parser = argparse.ArgumentParser(description='Run model vs STREME on exon data')
    parser.add_argument('--datadir', help = 'Path to the exon data', required = False, type = str,
                        default="/home/ebelm/genomegraph/data/241_species/20231123_subset150_NM_RefSeqBest/out_subset150_withEnforced_20_15_20_50_15_20_15_20_mammals/")
    parser.add_argument('--out', help = 'Output directory', required = True, type = str)
    parser.add_argument('--maxexons', help = 'Maximum number of exons to use', required = False, type = int)
    args = parser.parse_args()
    assert os.path.isdir(args.datadir), f"[ERROR] >>> Data dir {args.datadir} not found"
    if args.maxexons is not None:
        assert args.maxexons > 0, f"[ERROR] >>> Maximum number of exons must be positive, not {args.maxexons}"
        MAXEXONS = args.maxexons
    else:
        MAXEXONS = None

    outdir = args.out
    os.makedirs(outdir, exist_ok=True) # make sure that outdir exists

    # set logfile
    logging.basicConfig(filename = os.path.join(outdir, "logfile.txt"),
                        format="%(asctime)s %(levelname)s: %(message)s", 
                        encoding='utf-8', level=logging.DEBUG)

    logging.info("TensorFlow version: "+str(tf.__version__))

    # === LOAD DATA ===

    exonsetFiles = []
    warnings = []
    logging.info("[main] Getting exon list")
    for d in os.listdir(args.datadir):
        subdir = os.path.join(args.datadir, d)
        if re.match("exon_chr.+", d) and os.path.isdir(subdir):
            seqDataFile = os.path.join(subdir, "profile_finding_sequence_data.json")
            if not os.path.isfile(seqDataFile):
                warnings.append(f"[main] Expected file {seqDataFile} but not found, skipping")
                continue
                
            exonsetFiles.append(seqDataFile)

    logging.info(f"[main] {len(warnings)} warnings")
    for warning in warnings:
        logging.warning(warning)
        
    logging.info(f"[main] Number of exons: {len(exonsetFiles)}")
    
    # shuffle exon list
    random.shuffle(exonsetFiles)
    
    if MAXEXONS is not None and MAXEXONS < len(exonsetFiles):
        logging.info(f"[main] Limiting data to {MAXEXONS} exons")
        exonsetFiles = exonsetFiles[:MAXEXONS]
    
    # load sequences
    logging.info("[main] Loading sequences")
    allGenomes: list[SequenceRepresentation.Genome] = []
    singleExonGenomes: list[list[SequenceRepresentation.Genome]] = [] # run training on these
    stoi = {} # species to allGenomes list index
    for sl in exonsetFiles:
        sequences = SequenceRepresentation.loadJSONSequenceList(sl)

        # assert unique sequences and unique species (i.e. only one sequence per species)
        seqIDs = [seq.id for seq in sequences]
        species = [seq.species for seq in sequences]
        assert len(seqIDs) == len(set(seqIDs)), f"[ERROR] >>> Duplicate sequences in {sequences}"
        assert len(species) == len(set(species)), f"[ERROR] >>> Duplicate species in {sequences}"
        assert 'Homo_sapiens' in species, f"[ERROR] >>> Homo sapiens not in {sequences}"
        assert species[0] == 'Homo_sapiens', f"[ERROR] >>> Homo sapiens not first in {sequences}"

        skip = False
        for seq in sequences:
            assert seq.hasHomologies(), f"[ERROR] >>> Sequence {seq} has no homologies"
            assert len(seq.homology) == 1, f"[ERROR] >>> Sequence {seq} has not exactly one homology"

            if seq.species == 'Homo_sapiens':
                assert seq.elementsPossible(), f"[ERROR] >>> Sequence {seq} cannot have annotations"
                if len(seq.genomic_elements) == 0:
                    logging.warning(f"[main] Skipping sequence {seq} as it has no annotations")
                    skip = True

                break

        if skip:
            continue

        # create list of single-exon Genomes for training
        seGen = []
        for seq in sequences:
            # make annotations and transkripts unique
            SequenceRepresentation.makeAnnotationsUnique(seq)
            SequenceRepresentation.selectLongestTranscript(seq)

            g = SequenceRepresentation.Genome()
            g.addSequence(seq)
            seGen.append(g)

        singleExonGenomes.append(seGen)
        
        # store all sequences at their respective genomes
        for seq in sequences:
            if seq.species not in stoi:
                stoi[seq.species] = len(allGenomes) # add new genome, remember index
                allGenomes.append(SequenceRepresentation.Genome())

            allGenomes[stoi[seq.species]].addSequence(seq)
    
    logging.info("[main] Number of homologies: "+str(len(exonsetFiles)))
    logging.info("[main] Total number of genomes: "+str(len(allGenomes)))
    logging.info("[main] Total number of sequences: "+str(sum([len(g) for g in allGenomes])))
    logging.info("[main] Sequences per genome: "+\
                 ', '.join([f"{[len(g) for g in allGenomes].count(i)} genomes with {i} sequences"\
                              for i in range(max([len(g) for g in allGenomes])) \
                              if [len(g) for g in allGenomes].count(i) > 0]))

    logging.info("[main] Check unique annotations: "+str(checkUniqueAnnotations(allGenomes)))

    # store all genomes for later evaluation
    with open(os.path.join(outdir, "allGenomes.json"), 'wt') as fh:
        json.dump([g.toDict() for g in allGenomes], fh)

    # === TRAINING ===

    logging.info("[main] Starting training and evaluation")

    # some global settings
    genome_limit = 50
    
    # training iteration
    evaluator = training.MultiTrainingEvaluation()
    streme_evaluator = training.MultiTrainingEvaluation()
    for i, seGenomes in tqdm(enumerate(singleExonGenomes)):
        runID = f"{i:04}" # 0000, 0001, ...
        starttime = time()
        logging.info(f"[main] Start training and evaluation for run {runID}")

        seGenomes = seGenomes[:min(len(seGenomes), genome_limit)]
        # store single-exon genomes for later evaluation
        with open(os.path.join(outdir, f"{runID}_singleExonGenomes.json"), 'wt') as fh:
            json.dump([g.toDict() for g in seGenomes], fh)

        # --- train our model ---
        logging.info(f"[main] Start training and evaluation on model for {runID}")
        data = ProfileFindingSetup.ProfileFindingDataSetup('real')
        data.addGenomes(seGenomes, verbose=True)
        trainsetup = ProfileFindingSetup.ProfileFindingTrainingSetup(data,
                                                                     tiles_per_X = 7,
                                                                     tile_size = 334,
                                                                     U = 200, n_best_profiles=1)
        trainsetup.initializeProfiles(enforceU=False, plot=False, overlapTilesize=6)
        try:
            training.trainAndEvaluate(runID, trainsetup, evaluator, 
                                      outdir, outprefix=f"{runID}_", 
                                      trainingWithReporting=True, rand_seed=SEED)
        except Exception as e:
            logging.error(f"[main] trainAndEvaluate failed for homology {i}, check log for details")
            logging.error(f"[main] Error message: {e}")
            continue

        evaluator.dump(os.path.join(outdir, "evaluator.json")) # save after each run

        # --- run STREME ---
        logging.info(f"[main] Start training and evaluation on STREME for {runID}")
        streme_wd = os.path.join(outdir, f"{runID}_STREME")
        streme_runner = Streme.Streme(working_dir = streme_wd,
                                      load_streme_script= "/home/ebelm/Software/load_MEME.sh")
        translated_seqs = []
        for seGenome in seGenomes:
            for sequence in seGenome:
                for frame in range(6):
                    translated_seqs.append(SequenceRepresentation.TranslatedSequence(sequence, frame, 
                                                                                     replaceSpaceWithX=True))
                
        SequenceRepresentation.sequenceListToFASTA(translated_seqs, os.path.join(streme_wd, "data.fasta"))

        try:
            _ = streme_runner.run(runID, translated_seqs, seGenomes, streme_evaluator, verbose=True, plot_links=True)
        except Exception as e:
            logging.error(f"[main] STREME failed for homology {i}, check log for details")
            logging.error(f"[main] Error message: {e}")
            continue
        
        streme_evaluator.dump(os.path.join(outdir, "streme_evaluator.json")) # save after each run

        endtime = time()
        runtime = endtime - starttime
        logging.info(f"[main] Finished training and evaluation for {runID}. Took {runtime:.2f}s")



if __name__ == "__main__":
    main()