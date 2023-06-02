from dataclasses import dataclass
import json
import os

import numpy as np

import dataset as dsg
import SequenceRepresentation as sr
import sequtils as su

@dataclass
class ProfileFindingSetup:
    mode: str # one of ['toy', 'sim' or 'real']
    N: int = None
    genomes: list[sr.Genome] = None
    Q: np.ndarray = None
        
    # these fields only relevant for toy data
    genomeSizes: list[list[int]] = None
    insertPatterns: list[str] = None
    repeatPatterns: list[str] = None
    repeatsPerGenome: list[int] = None
    multiplyRepeats: list[int] = None
    mutationProb: float = 0.0
        
    # these fields only relevant for simulated data
    seqlen: int = 110000
    genelen: int = 140
    cdist: float = 0.05
    ncdist: float = 0.1
    tree: str = 'star'
    omega: float = 0.4
    
    def __post_init__(self):
        assert self.mode in ['toy', 'sim', 'real'], \
            f"[ERROR] >>> mode must be one of ['toy', 'sim' or 'real'], not {self.mode}"
        
        # set some defaults based on how self was constructed
        if self.mode != 'toy':
            self.genomeSizes = None
            self.insertPatterns = None
            self.repeatPatterns = None
            self.repeatsPerGenome = None
            self.multiplyRepeats = None
            self.mutationProb = None
        else: # set defaults if not yet set
            if self.N is None:
                self.N = 8
            if self.genomeSizes is None:
                self.genomeSizes = [[10000]] * self.N
            if self.insertPatterns is None:
                self.insertPatterns = ["ATGGCAAGAATTCAATCTACTGCAAATAAAGAA"]
            if self.repeatPatterns is None:
                self.repeatPatterns = ['AGAGAACCTGAAGCTACTGCTGAACCTGAAAGA']
            if self. repeatsPerGenome is None:
                self.repeatsPerGenome = [10]
            if self.multiplyRepeats is None:
                self.multiplyRepeats = [1]
                
        if self.mode != 'sim':
            self.seqlen = None
            self.genelen = None
            self.cdist = None
            self.ncdist = None
            self.tree = None
            self.omega = None
        else:
            if self.N is None:
                self.N = 80
            
                
            
    def addGenomes(self, genomes: list[sr.Genome] = None, verbose: bool =True):
        """ Add genomes either directly or by generating them
            Arguments:
              genomes: str or list[SequenceRepresentation.Genome]  Path to a JSON of a list of Genome objects or list of
                                                                     Genome objects to add. If None (default), generate 
                                                                     genomes.
              verbose: bool                                        Print more information on Q and on generating toy 
                                                                     data if true
        """
        if genomes is not None:
            assert type(genomes) is list or type(genomes) is str, "[ERROR] >>> genomes must be a str or a list of " \
                                                         + f"SequenceRepresentation.Genome objects, not {type(genomes)}"
            if type(genomes) is str:
                assert os.path.isfile(genomes), f"[ERROR] >>> Path `{genomes}` does not exist or is not a valid file"
                # expecting a list of list of dicts of SequencRepresentation.Sequences
                with open(genomes, "rt") as fh:
                    genomesreps = json.load(fh)
                    
                self.genomes = []
                for genomerep in genomesreps:
                    genome = sr.Genome()
                    for seqdict in genomerep:
                        genome.addSequence(sr.fromJSON(jsonstring = json.dumps(seqdict)))
                        
                    self.genomes.append(genome)
                    
            else:
                assert len(genomes) > 0, "[ERROR} >>> genomes must contain genomes, empty list not allowed"
                for i, genome in enumerate(genomes):
                    assert type(genome) is sr.Genome, "[ERROR] >>> Elements of genomes must be " \
                        + f"SequenceRepresentation.Genome objects, but instance {i} is of type {type(genome)}"

                self.genomes = genomes
                
            self.N = len(genomes)
            
        else: 
            assert self.N is not None and self.N//1 == self.N and self.N >= 1, \
                f"[ERROR] >>> N must be an integer >= 1, not {self.N}"
            if self.mode == 'toy':
                assert self.genomeSizes is not None, "[ERROR] >>> genomeSizes must be set"
                self.genomes = dsg.getRandomGenomes(self.N, self.genomeSizes, self.insertPatterns, self.repeatPatterns,
                                                    self.mutationProb, self.multiplyRepeats, self.repeatsPerGenome, 
                                                    verbose)
            elif self.mode == 'sim':
                self.genomes = dsg.simulateGenomes(self.N, self.seqlen, self.genelen, self.cdist, self.ncdist, 
                                                   self.tree, self.omega)
                
        # set Q, overwrites anything that was initialized!
        if self.Q is not None:
            print("[WARNING] >>> Overwriting previous Q:", self.Q)
            
        if self.mode == 'toy':
            self.Q = np.ones(21, dtype=np.float32)/21
        else:
            self.Q = dsg.backGroundAAFreqs(self.extractSequences(), verbose)
            
            
            
    def expectedPatterns(self):
        """ Only in toy mode, prints and returns expected insert and repeat patterns as AA sequences"""
        if self.mode == 'toy':
            desiredPatternAA = []
            for pattern in self.insertPatterns:
                desiredPatternAA.extend(su.six_frame_translation(pattern))

            print("Desired:", desiredPatternAA)

            repeatPatternAA = []
            for pattern in self.repeatPatterns:
                repeatPatternAA.extend(su.six_frame_translation(pattern))

            print("Repeat:", repeatPatternAA)

            return desiredPatternAA, repeatPatternAA
        else:
            print(f"[WARNING] >>> expectedPatterns() only valid in `toy` mode, not in `{self.mode}` mode.")
            return None, None
        
        
        
    def extractSequences(self):
        """ Returns the DNA sequences in a list of lists, outer list for the genomes, 
              inner lists for the sequences in the genomes. """
        assert self.genomes is not None, "[ERROR] >>> Add genomes first"
        sequences = []
        #seqnames = []
        for genome in self.genomes:
            sequences.append([])
            #seqnames.append([])
            for sequence in genome:
                sequences[-1].append(sequence.getSequence(sequence.strand == '-'))
                #seqnames[-1].append(sequence.id)
                
        return sequences#, seqnames
        
        
        
    def storeGenomes(self, filename: str, overwrite = False):
        if os.path.isfile(filename):
            if overwrite:
                print(f"[INFO] >>> Overwriting {filename}")
            else:
                assert not os.path.isfile(filename), f"[ERROR] >>> Overwriting {filename} not allowed"
                
        with open(filename, "wt") as fh:
            # generates a list of list of dicts of SequencRepresentation.Sequences
            json.dump([genome.toDict() for genome in self.genomes], fh) 