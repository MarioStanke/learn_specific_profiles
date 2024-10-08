from dataclasses import dataclass
import logging
import numpy as np
import random

from . import Links
from . import ModelDataSet
from . import plotting
from .utils import oneHot

# set logging level for logomaker to avoid debug message clutter
logging.getLogger('plotting.logomaker').setLevel(logging.WARNING)


@dataclass
class ProfileFindingTrainingSetup:
    """ Dataclass to store the parameters for training a profile finding model """

    data: ModelDataSet.ModelDataSet
    
    U: int    # number of profiles to train
    k: int    # length of profiles
    midK: int # length of kmers to intialize middle part of profiles
    s: int    # profile shift to both sides
    
    epochs: int  # max number of epochs to train until force report
    gamma: float # softmax scale (used in 'experiment' loss function)
    l2: float    # L2 reg factor
    match_score_factor: float
    learning_rate: float 
    lr_patience: int # number of epochs to wait for loss decrease before trigger learning rate reduction
    lr_factor: float # factor to reduce learning rate by
    rho: float       # influence of initial sampling position on profile initialization via seeds
    sigma: float     # stddev of random normal values added to profile initialization via seeds (mean 0)
    profile_plateau: int       # number of epochs to wait for loss plateau to trigger profile reporting
    profile_plateau_dev: float # upper threshold for stddev of loss plateau to trigger profile reporting

    n_best_profiles: int # number of best profiles to report
    
    # use prior knowledge on amino acid similarity
    phylo_t: float # values in [0, 250] are reasonable (0.0 means no prior knowledge)
                   # time a CTMC evolves from the parameter profile P to the profile
                   # that is used for scoring/searching
                   # if t==0.0 this prior knowledge is not used
                   # requires amino acid alphabet, in particular k=20

    # TODO: deprecated, remove in future
    alpha: float = 1e-6 # loss norm (deprecated, not used anymore)
    lossStrategy: str = 'experiment' #'score' #'experiment' #'softmax' (deprecated)

    def __post_init__(self):
        assert self.U > 0, f"[ERROR] >>> {self.U=} must be > 0"
        assert self.k > 0, f"[ERROR] >>> {self.k=} must be > 0"
        assert self.midK > 0, f"[ERROR] >>> {self.midK=} must be > 0"
        assert self.k >= self.midK, f"[ERROR] >>> {self.k=} must be >= {self.midK=}"
        assert self.s >= 0, f"[ERROR] >>> {self.s=} must be >= 0"
        assert self.epochs > 0, f"[ERROR] >>> {self.epochs=} must be > 0"
        assert self.n_best_profiles > 0, f"[ERROR] >>> {self.n_best_profiles=} must be > 0"

        genome_sizes = [sum([len(s[0]) for s in genome]) for genome in self.data.getRawData()]
        # one training step: using batch_size*tiles_per_X*tile_size characters from each genome,
        # thus estimate an epoch via the mean genome size (sum of sequence lengths) and the characters used per step
        steps_per_epoch = max(1, 
                              np.mean(genome_sizes) // (self.data.batch_size*self.data.tiles_per_X*self.data.tile_size))
        self.steps_per_epoch = steps_per_epoch
        self.initProfiles: np.ndarray = None # type: ignore
        self.initKmerPositions: dict[str, list[Links.Occurrence]] = {}
        self.trackProfiles: list[int] = []



    def initializeProfiles_kmers(self, enforceU = True, minU = 10, minOcc = 8, overlapTilesize = 6, plot = False):
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

        rawdata = self.data.getRawData()
        alphabet = self.data.alphabet
        
        # count kmers
        kmerToOcc : dict[str, list[tuple]] = {}
        seenTiles = set() # store position hashes of tiles that have been seen
        for g in range(len(rawdata)):
            for c in range(len(rawdata[g])):
                for f in range(len(rawdata[g][c])):
                    seq = rawdata[g][c][f]
                    for i in range(len(seq)-self.midK+1):
                        kmer = seq[i:i+self.midK]
                        # if (' ' not in kmer): # skip unknown AAs (i.e. unknown codons)
                        if all([x in alphabet for x in kmer]):                            
                            tile = (g, c, f, i//overlapTilesize)
                            if (kmer not in kmerToOcc) and (tile in seenTiles):
                                continue # ignore new kmers that overlap with already seen kmers

                            if kmer not in kmerToOcc:
                                kmerToOcc[kmer] = [] # store occs

                            kmerToOcc[kmer].append((g,c,f,i))
                            seenTiles.add(tile)

        kmerCount = [(kmer, len(kmerToOcc[kmer])) for kmer in kmerToOcc]
        kmerCount.sort(key=lambda t: t[1], reverse=True)
                            
        # initialize all profiles with most frequent kmers
        if enforceU:
            if len(kmerToOcc) < self.U:
                logging.warning("[ProfileFindingSetup.ProfileFindingTrainingSetup.initializeProfiles] >>> Only " + \
                                f"{len(kmerToOcc)} different kmers found, but {self.U} profiles requested. " + \
                                "Using all kmers plus random kmers.")
                midKmers = list(kmerToOcc.keys())
                # add randomly generated kmers
                randKmers = [''.join(random.choices(list(alphabet), list(self.data.Q), k=self.midK)) \
                                for _ in range(self.U-len(kmerToOcc))]
                midKmers.extend(randKmers)
            else:
                midKmers = [t[0] for t in kmerCount[:self.U]]

            assert len(midKmers) == self.U, f"[ERROR] >>> {len(midKmers)} != {self.U}"
            self.trackProfiles = list(range(len(midKmers)))

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

                self.trackProfiles = list(range(len(midKmers)))
        
        self.initKmerPositions = {}
        for kmer in midKmers:
            occtuples = kmerToOcc[kmer]
            kmerOccs = ModelDataSet.siteConversionHelper(self.data, occtuples, self.midK)
            self.initKmerPositions[kmer] = kmerOccs # kmerToOcc[kmer]
            
        logging.info("[ProfileFindingSetup.ProfileFindingTrainingSetup.initializeProfiles] >>> Number of profiles: " + \
                     str(len(midKmers)))
        if len(midKmers) != self.U:
            logging.warning("[ProfileFindingSetup.ProfileFindingTrainingSetup.initializeProfiles] >>> " + \
                            f"{len(midKmers)} profiles initialized, but {self.U} requested. " + \
                            f"Resetting U to {len(midKmers)}")
            self.U = len(midKmers)


        if len(self.trackProfiles) == 0:
            self.trackProfiles = list(range(len(midKmers)))

        self.initProfiles = getCustomMidProfiles(midKmers, alphabet, self.k+(2*self.s), self.data.Q, 
                                                 mid_factor=4, bg_factor=1)
        
        if self.n_best_profiles > self.initProfiles.shape[2]:
            logging.warning("[ProfileFindingSetup.ProfileFindingTrainingSetup.initializeProfiles] >>> " + \
                            f"n_best_profiles ({self.n_best_profiles}) > number of profiles " + \
                            f"({self.initProfiles.shape[2]}), setting n_best_profiles to {self.initProfiles.shape[2]}")
            self.n_best_profiles = self.initProfiles.shape[2]

        if plot:
            # softmax to see the profiles as they would be reported by the model
            softmaxProfiles = np.transpose(np.exp(self.initProfiles), (1,2,0)) # axis one needs to become 0 for softmax
            softmaxProfiles = softmaxProfiles / np.sum(softmaxProfiles, axis=0)
            softmaxProfiles = np.transpose(softmaxProfiles, (2,0,1))    
            plotting.plotLogo(softmaxProfiles, alphabet)
    


    def initializeProfiles_seeds(self, rand_seed: int = None): # type: ignore
        """ Used to be model.seed_P_genome. Initializes the profiles with seeds from the genomes. 
            Sets self.initProfiles. Sets self.trackProfiles to all profiles."""
        import tensorflow as tf

        flatg = []
        for seqs in self.data.getRawData():
            for frameseqs in seqs:
                flatg.extend(frameseqs)

        lensum = sum([len(s) for s in flatg])
        weights = [len(s)/lensum for s in flatg]
        weights = tf.nn.softmax(weights).numpy() # type: ignore
        nprng = np.random.default_rng(rand_seed)
        seqs = nprng.choice(len(flatg), self.U, replace=True, p=weights)
        ks = self.k + (2*self.s) # trained profile width (k +- shift)

        oneProfile_logit_like_Q = np.log(self.data.Q)
        P_logit_init = np.zeros((ks, self.data.alphabet_size(), self.U), dtype=np.float32)
        for j in range(self.U):
            i = seqs[j] # seq index
            assert len(flatg[i]) > ks, str(len(flatg[i]))
            pos = nprng.choice(len(flatg[i])-ks, 1)[0]
            seedseq = flatg[i][pos:pos+ks]
            OH = oneHot(seedseq, self.data.alphabet)
            assert OH.shape == (ks, self.data.alphabet_size()), f"{OH.shape} != {(ks, self.data.alphabet_size())}"
            seed = self.rho * OH + oneProfile_logit_like_Q + nprng.normal(scale=self.sigma, size=OH.shape)
            P_logit_init[:,:,j] = seed

        self.initProfiles = P_logit_init
        self.trackProfiles = list(range(self.U))


    
    def print(self):
        """ Print all members except the _genomes and data.genomes to keep it readable. """
        # self.data.print("data.")
        for attr, value in self.__dict__.items():
            if attr != "data":
                print(attr, "=", value)



# General functions

def getCustomMidProfiles(midSeqs: list[str], alphabet: list[str], k: int, Q: np.ndarray, 
                         mid_factor: float = 1, bg_factor: float = 0):
    """ Generate profiles in which the middle positions are based on a kmer. 
    
    Args:
        midSeqs: list of kmer strings
        alphabet: list of allowed characters in the sequences
        k: length of the profiles
        Q: background distribution
        mid_factor: scaling factor for the middle positions
        bg_factor: scaling factor for the background positions

    Returns:
        profiles: np.ndarray of shape (k, len(alphabet), len(midSeqs))
    """

    assert max([len(m) for m in midSeqs]) <= k, "[ERROR] >>> mid element lengths cannot exceed k"
    assert len(Q) == len(alphabet), f"[ERROR] >>> Q must have {len(alphabet)} elements"
    assert mid_factor > 0, "[ERROR] >>> mid_factor must be > 0"
    assert bg_factor >= 0, "[ERROR] >>> bg_factor must be >= 0"
    
    U = len(midSeqs)
    profiles = np.repeat([Q], repeats=k, axis=0)        # -> (k, alphabet_size)
    profiles = np.repeat([profiles], repeats=U, axis=0) # -> (U, k, alphabet_size)
    profiles = np.transpose(profiles, (1,2,0))          # -> (k, alphabet_size, U)
    
    for u in range(U):
        midlen = len(midSeqs[u])
        lflank = (k-midlen)//2
        bgmid = np.repeat([Q], repeats=midlen, axis=0) * bg_factor # scaled background for middle positions
        mid = oneHot(midSeqs[u], alphabet) * mid_factor                     # scaled kmer for middle positions
        profiles[lflank:lflank+midlen,:,u] = (mid + bgmid)
                
    return profiles