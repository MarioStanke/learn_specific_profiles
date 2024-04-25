from dataclasses import dataclass
import json
import logging
import numpy as np
import os
import random

from . import dataset
from . import plotting
from . import SequenceRepresentation as sr
from . import sequtils as su

# set logging level for logomaker to avoid debug message clutter
logging.getLogger('plotting.logomaker').setLevel(logging.WARNING)


@dataclass
class ProfileFindingDataSetup:
    """ Class to hold all information and methods needed to generate a dataset for profile finding.
        Can be used in one of three modes: 'toy', 'sim' or 'real'. The appropriate defaults are set accordingly
        but can be changed by setting the respective fields. """
    
    mode: str # one of ['toy', 'sim' or 'real']
    N: int = None
    genomes: list[sr.Genome] = None
    Q: np.ndarray = None
        
    # these fields only relevant for toy data
    toy_genomeSizes: list[list[int]] = None
    toy_insertPatterns: list[str] = None
    toy_repeatPatterns: list[str] = None
    toy_repeatsPerGenome: list[int] = None
    toy_multiplyRepeats: list[int] = None
    toy_mutationProb: float = 0.0
        
    # these fields only relevant for simulated data
    sim_seqlen: int = 110000
    sim_genelen: int = 140
    sim_cdist: float = 0.05
    sim_ncdist: float = 0.1
    sim_tree: str = 'star'
    sim_omega: float = 0.4
    
    def __post_init__(self):
        assert self.mode in ['toy', 'sim', 'real'], \
            f"[ERROR] >>> mode must be one of ['toy', 'sim' or 'real'], not {self.mode}"
        
        # set some defaults based on how self was constructed
        if self.mode != 'toy':
            self.toy_genomeSizes = None
            self.toy_insertPatterns = None
            self.toy_repeatPatterns = None
            self.toy_repeatsPerGenome = None
            self.toy_multiplyRepeats = None
            self.toy_mutationProb = None
        else: # set defaults if not yet set
            if self.N is None:
                self.N = 8
            if self.toy_genomeSizes is None:
                self.toy_genomeSizes = [[10000]] * self.N
            if self.toy_insertPatterns is None:
                self.toy_insertPatterns = ["ATGGCAAGAATTCAATCTACTGCAAATAAAGAA"]
            if self.toy_repeatPatterns is None:
                self.toy_repeatPatterns = ['AGAGAACCTGAAGCTACTGCTGAACCTGAAAGA']
            if self. toy_repeatsPerGenome is None:
                self.toy_repeatsPerGenome = [10]
            if self.toy_multiplyRepeats is None:
                self.toy_multiplyRepeats = [1]
                
        if self.mode != 'sim':
            self.sim_seqlen = None
            self.sim_genelen = None
            self.sim_cdist = None
            self.sim_ncdist = None
            self.sim_tree = None
            self.sim_omega = None
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
                assert self.toy_genomeSizes is not None, "[ERROR] >>> genomeSizes must be set"
                self.genomes = dataset.getRandomGenomes(self.N, self.toy_genomeSizes, self.toy_insertPatterns, 
                                                        self.toy_repeatPatterns, self.toy_mutationProb, 
                                                        self.toy_multiplyRepeats, self.toy_repeatsPerGenome, verbose)
            elif self.mode == 'sim':
                self.genomes = dataset.simulateGenomes(self.N, self.sim_seqlen, self.sim_genelen, self.sim_cdist, 
                                                       self.sim_ncdist, self.sim_tree, self.sim_omega)
                
        # set Q, overwrites anything that was initialized!
        if self.Q is not None:
            logging.warning("[ProfileFindingSetup.ProfileFindingDataSetup.addGenomes] >>> Overwriting previous Q: " + \
                            str(self.Q))
            
        if self.mode == 'toy':
            self.Q = np.ones(21, dtype=np.float32)/21
        else:
            self.Q = dataset.backGroundAAFreqs(self.extractSequences(), verbose)
            
            
            
    def expectedPatterns(self):
        """ Only in toy mode, prints and returns expected insert and repeat patterns as AA sequences"""
        if self.mode == 'toy':
            desiredPatternAA = []
            for pattern in self.toy_insertPatterns:
                desiredPatternAA.extend(su.six_frame_translation(pattern))

            print("Desired:", desiredPatternAA)

            repeatPatternAA = []
            for pattern in self.toy_repeatPatterns:
                repeatPatternAA.extend(su.six_frame_translation(pattern))

            print("Repeat:", repeatPatternAA)

            return desiredPatternAA, repeatPatternAA
        else:
            logging.warning("[ProfileFindingSetup.ProfileFindingDataSetup.expectedPatterns] >>> expectedPatterns() " + \
                            f"only valid in `toy` mode, not in `{self.mode}` mode.")
            return None, None
        
        
        
    def extractSequences(self, nonePadding=False):
        """ Returns the DNA sequences in a list of lists, outer list for the genomes, 
              inner lists for the sequences in the genomes. 
            Always returns the sequences on the positive strand, since six-frame-translation is performed in dataset
              creation anyway. 
              
            If nonePadding is True, the inner lists are padded with None to the length of the longest inner list in the
              genomes list."""
        assert self.genomes is not None, "[ERROR] >>> Add genomes first"
        sequences = []
        for genome in self.genomes:
            sequences.append([])
            for sequence in genome:
                sequences[-1].append(sequence.getSequence(rc = False))

        # For dataset creation, uneven genomes list of contigs is not supported as tf apparently fails to handle ragged
        # tensors there. Thus, do a "None-padding" of the genomes list
        if nonePadding:
            maxLen = max([len(g) for g in sequences])
            for g in range(len(sequences)):
                while len(sequences[g]) < maxLen:
                    sequences[g].append("")
                
        return sequences
    


    def print(self, prefix = ''):
        """ Print all members except genomes to keep it readable """
        for attr, value in self.__dict__.items():
            if attr != 'genomes':
                print(prefix+attr, '=', value)


    
    def stats(self):
        """ Print some 'statistics' about the genomes """
        print("Number of genomes:", len(self.genomes))
        nseqs = sum([len(g) for g in self.genomes])
        print("Number of sequences:", nseqs)
        print("Average number of sequences per genome:", f"{nseqs/len(self.genomes):.2f}")
        seqlens = [sum([len(s) for s in g]) for g in self.genomes]
        print("Total sequence length:", sum(seqlens))
        print("Average sequence length:", f"{np.mean(seqlens):.2f}")
        print("Average sequence length per genome:", f"{np.mean(seqlens)/len(self.genomes):.2f}")
        print("Average sequence length per sequence:", f"{np.mean(seqlens)/nseqs:.2f}")
        
        
        
    def storeGenomes(self, filename: str, overwrite = False):
        if os.path.isfile(filename):
            if overwrite:
                logging.info(f"[ProfileFindingSetup.ProfileFindingDataSetup.storeGenomes] >>> Overwriting {filename}")
            else:
                assert not os.path.isfile(filename), f"[ERROR] >>> Overwriting {filename} not allowed"
                
        with open(filename, "wt") as fh:
            # generates a list of list of dicts of SequencRepresentation.Sequences
            json.dump([genome.toDict() for genome in self.genomes], fh) 



@dataclass
class ProfileFindingTrainingSetup:
    """ Dataclass to store the parameters for training a profile finding model """

    data: ProfileFindingDataSetup
    tile_size: int = 334  # tile size measured in amino acids
    batch_size: int = 1   # number of X to generate per batch
    tiles_per_X: int = 13 # number of tiles per X (-> X.shape[0])
    prefetch: int = 3     # number of batches to prefetch
    
    U: int = 200        # number of profiles to train
    k: int = 20         # length of profiles
    midK: int = 12      # length of kmers to intialize middle part of profiles
    s: int = 0          # profile shift to both sides
    
    epochs: int = 350   # number of epochs to train
    alpha: float = 1e-6 # loss norm (deprecated, not used anymore)
    gamma: float = 1    # softmax scale (used in 'experiment' loss function)
    l2: float = 0.01    # L2 reg factor
    match_score_factor: float = 0.7
    learning_rate: float = 2 
    lr_patience: int = 5    # number of epochs to wait for loss decrease before trigger learning rate reduction
    lr_factor: float = 0.75 # factor to reduce learning rate by
    rho: float = 0      # influence of initial sampling position on profile initialization via seeds
    sigma: float = 1    # stddev of random normal values added to profile initialization via seeds (mean 0)
    profile_plateau: int = 10        # number of epochs to wait for loss plateau to trigger profile reporting
    profile_plateau_dev: float = 150 # upper threshold for stddev of loss plateau to trigger profile reporting

    n_best_profiles: int = 2 # number of best profiles to report
    lossStrategy: str = 'experiment' #'score' #'experiment' #'softmax'

    # use prior knowledge on amino acid similarity
    phylo_t = 0.0 # values in [0, 250] are reasonable (0.0 means no prior knowledge)
    # time a CTMC evolves from the parameter profile P to the profile
    # that is used for scoring/searching
    # if t==0.0 this prior knowledge is not used
    # requires amino acid alphabet, in particular k=20


    # do not set manually, gets calculated automatically
    steps_per_epoch: int = None
    initProfiles: np.ndarray = None
    trackProfiles: list = None
    initKmerPositions: dict = None
    _genomes: list[list[str]] = None # A persistent copy of the genomes that can be mutated during training.
                                     # Needed because repeated calls to data.extractSequences() would not work as
                                     # expected since it creates a new data structure with strings that are not
                                     # mutable and get replaced by new objects when altered, thus the underlying list of
                                     # SequenceRepresentation.Sequence objects would not be altered and changes are lost
                                     # after the next call to data.extractSequences().

    def __post_init__(self):
        genome_sizes = [sum([len(s) for s in genome]) for genome in self.data.genomes]
        steps_per_epoch = max(1, np.mean(genome_sizes) // (self.batch_size*self.tiles_per_X*self.tile_size*3))
        if self.steps_per_epoch is not None:
            #print(f"[WARNING] >>> Overwriting steps_per_epoch ({self.steps_per_epoch}) with {steps_per_epoch}")
            logging.warning("[ProfileFindingSetup.ProfileFindingTrainingSetup.__post_init__] >>> Overwriting " + \
                            f"steps_per_epoch ({self.steps_per_epoch}) with {steps_per_epoch}")

        self.steps_per_epoch = steps_per_epoch
        self._genomes = self.data.extractSequences(nonePadding=True)



    def initializeProfiles(self, enforceU = True, minU = 10, minOcc = 8, overlapTilesize = 6, plot = False):
        """ Initializes the profiles with most frequent kmers in the genomes. If not enough kmers are found, additional
            random kmers are added. If enforceU is True, exactly U profiles are initialized. If enforceU is False, at
            least minU profiles are initialized, starting with the most frequent kmers that occur at least minOcc times.
            All kmers that are equally frequent are included, except when U would be exceeded.
            Ignores kmers that overlap with already seen kmers by at most overlapTilesize.
            Sets self.initProfiles. Sets self.trackProfiles, either to all profiles from kmers that exceed minOcc if not
            enforceU and there are any, or to all profiles in all other cases.
        
        Args:
            enforceU (bool, optional): If True, enforces that the number of profiles is exactly U. Defaults to True.
            minU (int, optional): Only if enforceU is False. Minimum number of profiles to initialize, starting with the
                                  most frequent kmers. Defaults to 10. At most U profiles are initialized.
            minOcc (int, optional): Only if enforceU is False. Minimum number of occurences of a kmer to be considered.
                                    Defaults to 8. Is ignored if minU would not be reached otherwise.
            overlapTilesize (int, optional): Maximum overlap of kmers to be ignored.
            plot (bool, optional): If True, plots the initialized profiles. Defaults to False.
        """

        assert minU <= self.U, f"[ERROR] >>> minU ({minU}) must be <= U ({self.U})"

        seenTiles = set() # store position hashes of tiles that have been seen
        # count kmers
        kmerToOcc = {}
        for g in range(len(self.data.genomes)):
            for c in range(len(self.data.genomes[g])):
                translated = su.six_frame_translation(self.data.genomes[g][c].getSequence())
                for f in range(len(translated)):
                    seq = translated[f]
                    for i in range(len(seq)-self.midK+1):
                        kmer = seq[i:i+self.midK]
                        if (' ' not in kmer): # skip unknown AAs (i.e. unknown codons)
                            pos = su.convert_six_frame_position(i, f, len(self.data.genomes[g][c]), 
                                                                dna_to_aa=False)
                            assert 0 <= pos and pos < len(self.data.genomes[g][c]), \
                                f"[ERROR] >>> {pos} is not in [0,{len(self.data.genomes[g][c])})" \
                                    +f"({self.data.genomes[g][c]})"
                            
                            tile = (g, c, pos//overlapTilesize)
                            if (kmer not in kmerToOcc) and (tile in seenTiles):
                                continue # ignore new kmers that overlap with already seen kmers
                            if kmer not in kmerToOcc:
                                kmerToOcc[kmer] = [] # store occs

                            kmerToOcc[kmer].append((g,c,pos))
                            seenTiles.add(tile)

        kmerCount = [(k, len(kmerToOcc[k])) for k in kmerToOcc]
        kmerCount.sort(key=lambda x: x[1], reverse=True)
        self.trackProfiles = [] 
                            
        # initialize all profiles with most frequent kmers
        if enforceU:
            if len(kmerToOcc) < self.U:
                logging.warning("[ProfileFindingSetup.ProfileFindingTrainingSetup.initializeProfiles] >>> Only " + \
                                f"{len(kmerToOcc)} different kmers found, but {self.U} profiles requested. " + \
                                "Using all kmers plus random kmers.")
                midKmers = list(kmerToOcc.keys())
                # add randomly generated kmers
                randKmers = [''.join(random.choices(list(su.aa_alphabet[1:]), list(self.data.Q), k=self.midK)) \
                                for _ in range(self.U-len(kmerToOcc))]
                midKmers.extend(randKmers)
            else:
                midKmers = [t[0] for t in kmerCount[:self.U]]

            assert len(midKmers) == self.U, f"[ERROR] >>> {len(midKmers)} != {self.U}"

        else:
            if len(kmerToOcc) < minU:
                logging.warning("[ProfileFindingSetup.ProfileFindingTrainingSetup.initializeProfiles] >>> Only " + \
                                f"{len(kmerToOcc)} different kmers found, but min. {minU} profiles requested. " + \
                                f"Violating minU and only initializing {len(kmerToOcc)} profiles.")
                midKmers = list(kmerToOcc.keys())
            else:
                midKmers = [t[0] for t in kmerCount if t[1] >= minOcc] # first kmers with at least minOcc occs
                self.trackProfiles = list(range(len(midKmers))) # track these profiles
                while len(midKmers) < minU:
                    if len(midKmers) >= len(kmerCount):
                        break

                    thresh = kmerCount[len(midKmers)][1] # next include kmers with at least thresh occs
                    moreKmers = [t[0] for t in kmerCount[len(midKmers):] if t[1] == thresh]
                    midKmers.extend(moreKmers)

                assert midKmers == [t[0] for t in kmerCount[:len(midKmers)]], \
                    f"[ERROR] >>> {midKmers} != {kmerCount[:len(midKmers)]}"

                if len(midKmers) > self.U:
                    midKmers = midKmers[:self.U]
        
        self.initKmerPositions = {}
        for kmer in midKmers:
            self.initKmerPositions[kmer] = kmerToOcc[kmer]
            
        logging.info("[ProfileFindingSetup.ProfileFindingTrainingSetup.initializeProfiles] >>> Number of profiles: " + \
                     str(len(midKmers)))
        if len(midKmers) != self.U:
            logging.warning("[ProfileFindingSetup.ProfileFindingTrainingSetup.initializeProfiles] >>> " + \
                            f"{len(midKmers)} profiles initialized, but {self.U} requested. " + \
                            f"Resetting U to {len(midKmers)}")
            self.U = len(midKmers)


        if len(self.trackProfiles) == 0:
            self.trackProfiles = list(range(len(midKmers)))

        self.initProfiles = getCustomMidProfiles(midKmers, self.k+(2*self.s), self.data.Q, mid_factor=4, bg_factor=1)
        
        if self.n_best_profiles > self.initProfiles.shape[2]:
            logging.warning("[ProfileFindingSetup.ProfileFindingTrainingSetup.initializeProfiles] >>> " + \
                            f"n_best_profiles ({self.n_best_profiles}) > number of profiles " + \
                            f"({self.initProfiles.shape[2]}), setting to {self.initProfiles.shape[2]}")
            self.n_best_profiles = self.initProfiles.shape[2]

        if plot:
            # softmax to see the profiles as they would be reported by the model
            softmaxProfiles = np.transpose(np.exp(self.initProfiles), (1,2,0)) # axis one needs to become 0 for softmax
            softmaxProfiles = softmaxProfiles / np.sum(softmaxProfiles, axis=0)
            softmaxProfiles = np.transpose(softmaxProfiles, (2,0,1))    
            plotting.plotLogo(softmaxProfiles)



    def getDataset(self,
                   repeat: bool = False,
                   withPosTracking: bool = False, 
                   tiles_per_X: int = None,
                   tile_size: int = None,
                   batch_size: int = None,
                   prefetch: int = None,
                   original_data: bool = False):
        """ Any 'None' argument specified here overwrites the defaults from object constuction. Uses the internal copy
            of the genomes so changes to self.genomes() will persist there, but not in the original genomes. 
            original_data: if True, the original, immutable genome data is used, otherwise (default) the internal copy
                           is used that can e.g. be softmasked during training and uppercased with genomeToAllUC(). """
        tilesPerX = tiles_per_X if tiles_per_X is not None else self.tiles_per_X
        tileSize = tile_size if tile_size is not None else self.tile_size
        batchSize = batch_size if batch_size is not None else self.batch_size
        prefetch_ = prefetch if prefetch is not None else self.prefetch
        ds = dataset.getDataset(self._genomes if not original_data else self.data.extractSequences(nonePadding=True),
                                tilesPerX,
                                tileSize,
                                withPosTracking)
        
        if repeat:
            ds = ds.repeat()
        
        if batchSize is not None:
            ds = ds.batch(batchSize)
            
        if prefetch_ is not None:
            ds = ds.prefetch(prefetch_)
            
        return ds
    


    def genomes(self):
        return self._genomes
    


    def genomeToAllUC(self):
        """ Make all bases in the genome upper case (in place!, useful after training with reporting). Works on the
            internal copy of the genomes, so the original genomes are not altered. """
        for g in range(len(self._genomes)):
            for c in range(len(self._genomes[g])):
                self._genomes[g][c] = self._genomes[g][c].upper()


    
    def print(self):
        """ Print all members except the _genomes and data.genomes to keep it readable. """
        self.data.print("data.")
        for attr, value in self.__dict__.items():
            if attr != "_genomes" and attr != "data":
                print(attr, "=", value)



# General functions

def getCustomMidProfiles(midSeqs: list[str], k: int, Q: np.ndarray, mid_factor: float = 1, bg_factor: float = 0):
    """ Generate profiles in which the middle positions are based on a kmer. 
    
    Args:
        midSeqs: list of kmer strings
        k: length of the profiles
        Q: background distribution
        mid_factor: scaling factor for the middle positions
        bg_factor: scaling factor for the background positions

    Returns:
        profiles: np.ndarray of shape (k, 21, U)
    """

    aa_alphabet = su.aa_alphabet[1:]
    assert max([len(m) for m in midSeqs]) <= k, "[ERROR] >>> mid element lengths cannot exceed k"
    assert len(Q) == len(aa_alphabet), "[ERROR] >>> Q must have "+str(len(aa_alphabet))+" elements"
    assert mid_factor > 0, "[ERROR] >>> mid_factor must be > 0"
    assert bg_factor >= 0, "[ERROR] >>> bg_factor must be >= 0"
    
    U = len(midSeqs)
    profiles = np.repeat([Q], repeats=k, axis=0)
    profiles = np.repeat([profiles], repeats=U, axis=0)
    profiles = np.transpose(profiles, (1,2,0))
    
    for u in range(U):
        midlen = len(midSeqs[u])
        lflank = (k-midlen)//2
        bgmid = np.repeat([Q], repeats=midlen, axis=0) * bg_factor # scaled background for middle positions
        mid = dataset.oneHot(midSeqs[u]) * mid_factor              # scaled kmer for middle positions
        profiles[lflank:lflank+midlen,:,u] = (mid + bgmid)
                
    return profiles