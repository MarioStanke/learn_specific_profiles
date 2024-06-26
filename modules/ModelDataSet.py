from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
import tensorflow as tf

from . import Links
from . import position_conversion as pc
from . import SequenceRepresentation as sr
from . import sequtils as su
from .typecheck import typecheck



_DNA_ALPHABET = ['A', 'C', 'G', 'T']
_TRANSLATED_ALPHABET = su.aa_alphabet[1:] # does not contain missing AA (' ')



@dataclass
class TileableSequence:
    """ Store a single sequence string and provide tiles of a given size. Makes data set generation way clearer. 
    The sequence is stored as a string and tiles are generated on the fly. Check with .finished if the sequence is
    exhausted. """
    seq: str
    tile_size: int
    reverse_tiling: bool = False # TODO: test is this influences model performance: iterate tiles from rc seqs / frame >=3 from back to front

    def __post_init__(self):
        self.finished = len(self.seq) == 0
        if not self.finished:
            self._start = 0 if not self.reverse_tiling else len(self.seq) - self.tile_size
            self._end = self._start + self.tile_size
            if self.reverse_tiling:
                assert self._end > 0
        else:
            self._start = 0
            self._end = 0
    
    # emulate a string that can be iterated over, checked for character membership, length, etc.
    def __contains__(self, c: str) -> bool:
        return c in self.seq
    
    def __getitem__(self, idx: int) -> str:
        return self.seq[idx]
    
    def __iter__(self):
        return iter(self.seq)

    def __len__(self) -> int:
        return len(self.seq)

    def __repr__(self) -> str:
        return self.seq
    
    def __str__(self) -> str:
        return self.seq
            
    def next_tile(self) -> tuple[str, int]:
        """ Returns the next tile of the sequence as string and the start position of that tile in the sequence. If the
        sequence is exhausted, returns an empty string and -1 and sets .finished to True. """
        if self.finished:
            return "", -1
        
        start = max(0, self._start) # start can be negative if reverse_tiling
        end = min(self._end, len(self.seq)) # end can be larger than len(seq) if not reverse_tiling
        tile = self.seq[start:end]

        # update tiles, mark as finished if necessary
        if not self.reverse_tiling:
            self._start += self.tile_size
            self._end += self.tile_size
            self.finished = self._start >= len(self.seq)

        else:
            self._start -= self.tile_size
            self._end -= self.tile_size
            self.finished = self._end <= 0

        return tile, start



def backgroundFreqs(sequences: list[list[list[str]]], alphabet: list[str], verbose: bool = False):
    """
    Commpute vector of background frequencies of letters in the sequences. Letters not in the alphabet are ignored.
    
        sequences: list of N lists of strings
    Returns:
        vector Q of shape len(alphabet)
    """
    Q = np.zeros(len(alphabet), dtype=np.float32)
    for genome in sequences:
        for seqs in genome:
            for seq in seqs:
                for c in seq:
                    if c in alphabet:
                        Q[alphabet.index(c)] += 1

    sum = Q.sum()
    if sum > 0:
        Q /= Q.sum()
    if verbose:
        logging.info(f"[ModelDataSet.backgroundFreqs] >>> background freqs: {sum} *")
        for c in range(len(alphabet)):
            logging.info(f"[ModelDataSet.backgroundFreqs] >>> {alphabet[c]} {Q[c]:.4f}")

    return Q



# Use a generator to get genome batches, simplified position handling
def createBatch(ntiles: int, tile_size: int, alphabet: list[str], frame_dimension_size: int,
                rawgenomes: list[list[list[str]]], reverse_tiling_ids: list[int],
                withPosTracking: bool = False):
    """ Generator function to create batches of tiles from a list of list of sequence strings. 
        Returns a tuple of (X, Y) where X is a numpy array of shape 
        (ntiles, N, frame_dimension_size, tile_size, alphabet_size) that contains the one-hot encoded tiles and Y is a
        numpy array of shape (ntiles, N, frame_dimension_size, 4) if `withPosTracking` was true (otherwise an empty 
        list) that contains the position information of the tiles in the genome. 
        
        Y[a,b,c,:] is a list of [genome_idx, sequence_idx, frame_idx, tile_start_position] for the tile `a` in 
        genome `b` at frame `c` in X.

        Set `withPosTracking` to true to be able to restore the position of a k-mer in the genome from the tile. """
    assert tile_size >= 1, f"[ERROR] >>> tile_size must be positive, non-zero (is: {tile_size})"

    # convert rawgenomes to TileableSequence objects
    rawgenomes = rawgenomes.astype('U')   # convert back from bytestring to unicode, otherwise one-hot does not work!
    alphabet = list(alphabet.astype('U')) 
    genomes = [ 
        [ 
            [ TileableSequence(seqs[f], tile_size, f in reverse_tiling_ids) for f in range(len(seqs)) ] \
                for seqs in genome 
        ] \
        for genome in rawgenomes
    ]

    #logging.debug(f"[ModelDataSet.createBatch] >>> {[[[len(f) for f in seq] for seq in genome] for genome in genomes]}")
    #logging.debug(f"[ModelDataSet.createBatch] >>> {genomes[0][0][0][:min(10, len(genomes[0][0][0]))]=}")
    #logging.debug(f"[ModelDataSet.createBatch] >>> {alphabet=}")

    N = len(genomes)
    state = []
    for seqs in genomes:
        if len(seqs) == 0:
            state.append({'sequence_idx': None, 'exhausted': True})
        else:
            state.append({'sequence_idx': 0, 
                          'exhausted': all([s.finished for s in seqs[0]]) if len(seqs) == 1 else False})

    while not all(s['exhausted'] for s in state):
        X = np.zeros([ntiles, N, frame_dimension_size, tile_size, len(alphabet)], dtype=np.float32)
        if withPosTracking:
            # [:,:,:,0] - genome idx, 
            # [:,:,:,1] - sequence idx, 
            # [:,:,:,2] - frame idx,
            # [:,:,:,3] - tile start position in the sequence, -1 if exhausted
            posTrack = np.ones([ntiles, N, frame_dimension_size, 4], dtype=np.int32) *-1
        else:
            posTrack = np.array([], dtype=np.int32)

        for t in range(ntiles):
            for g in range(N):
                if state[g]['exhausted']:
                    continue
                    
                sidx: int = state[g]['sequence_idx']
                assert sidx is not None
                sequences = genomes[g][sidx]
                assert len(sequences) == frame_dimension_size, f"[ERROR] >>> {len(sequences)} != {frame_dimension_size}"

                if withPosTracking:
                    posTrack[t,g,:,0] = g
                    posTrack[t,g,:,1] = sidx

                for f, seq in enumerate(sequences):
                    tileseq, start = seq.next_tile()
                    assert len(tileseq) <= tile_size, f"[ERROR] >>> {len(tileseq)} > {tile_size}, "
                    for i, c in enumerate(tileseq):
                        if c in alphabet:
                            X[t,g,f,i,alphabet.index(c)] = 1.0

                    if withPosTracking:
                        posTrack[t,g,f,2] = f
                        posTrack[t,g,f,3] = start

                    # initially had a bug with strings so that one-hot did not work. Does not apply for softmasked data.
                    if len(tileseq) > 0 and not (tileseq.lower() == tileseq):
                        if (X[t,g,f] == 0).all(): 
                            logging.warning(f"[ModelDataSet.createBatch] >>> {tileseq} {start} {tile_size} {len(tileseq)}")
                            logging.warning(f"[ModelDataSet.createBatch] >>> {tileseq[:min(10, len(tileseq))]=}")
                            logging.warning(f"[ModelDataSet.createBatch] >>> {X[t,g,f,:min(10, len(tileseq)),:]=}")
                            raise ValueError(f"Tile {t} in genome {g} at frame {f} was not one-hot encoded")

                if all([s.finished for s in sequences]):
                    state[g]['sequence_idx'] += 1
                    if state[g]['sequence_idx'] == len(genomes[g]):
                        state[g]['exhausted'] = True
                        state[g]['sequence_idx'] = None
                        
        yield X, posTrack



class DataMode(Enum):
    DNA = 1
    Translated = 2


class _TrainingDataWrapper:
    """ Takes care of providing the training data for a model. Depending on the datamode, the data is either used as
    genomic sequences or translated sequences. The data is stored as a list of Genomes and can be accessed as a list of
    list of list of strings (genomes[sequences[frames]]). The latter is used for training. The class also provides a 
    mapping of the training sequences to the sr.Sequence or sr.TranslatedSequence objects. 

    Note: data needs to be rectangular for tf, so smaller genomes are filled up with empty sequences internally
              such that all genomes have the same number of sequences. """
    def __init__(self, data: list[sr.Genome], datamode: DataMode, tile_size: int, replaceSpaceWithX: bool = False):
        assert datamode in DataMode, f"[ERROR] >>> datamode must be of type DataMode, not {type(datamode)}"
        self.datamode = datamode
        self._data = data
        self.tile_size = tile_size
        self.replaceSpaceWithX = replaceSpaceWithX
        trainingseqs, mapping = self._extract_training_sequences()
        self._training_sequences = trainingseqs
        self._sequence_mapping = mapping
        self.reverse_frame_ids = [1] if datamode == DataMode.DNA else [3,4,5] if datamode == DataMode.Translated else []


    def _extract_training_sequences(self) -> tuple[list[list[list[str]]], \
                                                   dict[int, dict[int, dict[int, sr.Sequence|sr.TranslatedSequence]]]]:
        """ Takes the original data and extracts the sequences as a list of lists of lists of strings.
         Depending on datamode, either uses the original genomic sequences and adds the respective reverse complements,
         or translates the sequences in 6 frames. Returns as second element a mapping of nested list indices to the 
         original genomic Sequence or TranslatedSequence objects in the data. 
         Filling up smaller genomes with empty sequences for training takes place here, 
         it is not reflected in the mapping. """
        max_genome_len = max(len(g) for g in self._data)
        if self.datamode == DataMode.DNA:
            training_sequences = [[[s.getSequence(), s.getSequence(rc=True)] for s in g] for g in self._data]
            sequence_mapping = {i: {j: {0: s} for j, s in enumerate(g)} for i, g in enumerate(self._data)}
            
        else:
            training_sequences = []
            sequence_mapping = {}
            for i, g in enumerate(self._data):
                sequences = []
                sequence_mapping[i] = {}
                for j, s in enumerate(g):
                    seqframes = []
                    sequence_mapping[i][j] = {}
                    for frame in range(6):
                        tseq = sr.TranslatedSequence(s, frame, self.replaceSpaceWithX)
                        sequence_mapping[i][j][frame] = tseq
                        seqframes.append(tseq.getSequence())

                    sequences.append(seqframes)

                training_sequences.append(sequences)

        # fill up with empty sequences as necessary
        for g in range(len(training_sequences)):
            if len(training_sequences[g]) < max_genome_len:
                n = max_genome_len - len(training_sequences[g])
                logging.debug(f"[ModelDataSet._extract_training_sequences] >>> Filling up genome {g} with {n} empty " \
                              + f"sequences to match length of {max_genome_len}")
                f = 2 if self.datamode == DataMode.DNA else 6
                for _ in range(n):
                    training_sequences[g].append([""] * f)
            
        return training_sequences, sequence_mapping
        

    def getGenomes(self) -> list[sr.Genome]:
        """ Returns the list of Genomes that the training data is based on. """
        return self._data
    

    def getTrainingData(self, fromSource: bool = False) -> list[list[list[str]]]:
        """ Returns a list of lists of lists of str with the (possibly translated, depends on datamode) 
        training data. Outer list: genomes/species, second list: sequences, inner list: frames ([fwd, rc] for DNA data,
        6 frames for translated data). If fromSource is False (default), uses the internal copy. Otherwise, a new 
        extraction from the untouched source data is returned. """
        return self._training_sequences if not fromSource else self._extract_training_sequences()[0]
    

    def getSequence(self, genome_idx: int, inner_idx: int, frame_idx: int = 0) -> sr.Sequence | sr.TranslatedSequence:
        """ Provided outer and inner list indices from the training data (as returned by getTrainingData()), returns
        the corresponding (Translated)Sequence object. If datamode is DNA, frame_idx must be 0 (it will not return a 
        modified Sequence object with the reverse complement sequence at frame 1!). """
        assert genome_idx in self._sequence_mapping, f"[ERROR] >>> No sequence with genome index {genome_idx}"
        assert inner_idx in self._sequence_mapping[genome_idx], f"[ERROR] >>> No sequence with inner index {inner_idx}"
        assert frame_idx in self._sequence_mapping[genome_idx][inner_idx], \
            f"[ERROR] >>> No sequence with frame index {frame_idx}"
        
        return self._sequence_mapping[genome_idx][inner_idx][frame_idx]
        


class ModelDataSet:
    """ 
    Class that provides the data for training and testing a Tensorflow model. 
    Data is always provided to this class as a list of Genomes, i.e. DNA-Data. It can be set via datamode that the
    model is trained on translated sequences. Translation and position-conversion is handled by this class. """

    def __init__(self, data: list[sr.Genome], datamode: DataMode, Q: np.ndarray = None,
                 tile_size: int = 334, tiles_per_X: int = 13, batch_size: int = 1, prefetch: int = 3,
                 replaceSpaceWithX: bool = False):
        """
        Note: data needs to be rectangular for tf, so smaller genomes are filled up with empty sequences internally
              such that all genomes have the same number of sequences. Keep this in mind when using .getRawData().
        Attributes:
            data: list of Genomes
            datamode: DataMode
            Q: np.ndarray -- Q-vector of background frequencies for the data, must be of same length as alphabet.size().
                             If None, it is calculated from the data
            tile_size: int -- size of the tiles during training
            tiles_per_X: int -- number of tiles per X
            batch_size: int -- number of batches to be used during training
            prefetch: int -- number of batches to prefetch
            replaceSpaceWithX: bool -- if True, replaces ' ' with 'X' in translated sequences, only needed in
                                        translated mode!
        """
        assert datamode in DataMode, f"[ERROR] >>> datamode must be of type DataMode, not {type(datamode)}"
        assert isinstance(data, list), f"[ERROR] >>> data must be of type list, not {type(data)}"
        assert len(data) > 1, f"[ERROR] >>> Need at least 2 genomes for training, not {len(data)}"
        assert all(typecheck(x, "Genome") for x in data), f"[ERROR] >>> data must be a list of Genomes"

        self.training_data = _TrainingDataWrapper(data, datamode, tile_size)
        self.alphabet = _DNA_ALPHABET if datamode == DataMode.DNA else _TRANSLATED_ALPHABET
        if Q is None:
            self.Q = backgroundFreqs(self.training_data.getTrainingData(), self.alphabet)

        assert self.Q.shape == (self.alphabet_size(),), \
            f"[ERROR] >>> Q-matrix must have shape ({self.alphabet_size()},), not {self.Q.shape}"
        
        self.tile_size = tile_size
        self.tiles_per_X = tiles_per_X
        self.batch_size = batch_size
        self.prefetch = prefetch
        

    def alphabet_size(self) -> int:
        return len(self.alphabet)


    def convertModelSites(self, sites: np.ndarray, sitelen: int = 1) -> list[Links.Occurrence]:
        """ Takes the sites tensor (as numpy array) from a SpecProModel.get_profile_match_sites() and converts them to
        a list of Links.Occurrence objects. This method takes care of position translation if DataMode.Translated.
        Returned Occurrences denote the leftmost position of the site in top strand coordinates relative to the DNA
        Sequence.
        
        Arguments:
            sites (np.ndarray): array of shape (X, 6) where X is the number of found sites and the second dimension
                                contains tuples with (genomeIdx, contigIdx, frameIdx, tileStartPos, tilePos, profileIdx)
            sitelen (int): length of the site, i.e the profile width, default is 1 (single position sites)
                                
        Returns:
            List of Links.Occurrences """
        # logging.debug(f"[ModelDataSet.convertModelSites] {self.training_data._sequence_mapping=}")
        assert len(sites.shape) == 2 and sites.shape[1] == 6, f"[ModelDataSet.convertModelSites] invalid {sites.shape=}"
        assert sitelen > 0, f"[ModelDataSet.convertModelSites] invalid {sitelen=}"
        occs = []
        for genomeIdx, contigIdx, frameIdx, tileStartPos, tilePos, profileIdx in sites:
            #logging.debug(f"[ModelDataSet.convertModelSites] {genomeIdx=} {contigIdx=} {frameIdx=} {tileStartPos=} " \
            #              + f"{tilePos=} {profileIdx=}")
            rawpos = int(tileStartPos+tilePos) # refers to the sequence at frameIdx, not necessarily the top strand!
            if self.training_data.datamode == DataMode.Translated:
                assert frameIdx in range(6), \
                    f"[ModelDataSet.convertModelSites] invalid {frameIdx=} for Translated DataMode"
                sequence: sr.Sequence = self.training_data.getSequence(genomeIdx, contigIdx, frameIdx).genomic_sequence
                if frameIdx < 3:
                    rc = False
                    dnapos = pc.aa_to_dna(frameIdx, rawpos) # convert to fwd DNA pos
                else: 
                    # site comes from rc dna, thus aa_to_dna of one-past-last _aa_ gives the one-past-last _dna_ pos on
                    #   rc strand. Subtract one to get the last dna pos on rc. In later conversion, this is the first
                    #   dna pos on the fwd stran! Also see test_position_conversion.py 
                    rc = True
                    aa_site_end = rawpos + sitelen
                    rc_site_end = pc.aa_to_dna(frameIdx-3, aa_site_end)
                    dnapos = rc_site_end - 1

                sitelen = sitelen * 3 # convert aa-site to dna-site length

            else:
                assert frameIdx in [0,1], f"[ModelDataSet.convertModelSites] invalid {frameIdx=} for DNA DataMode"
                sequence: sr.Sequence = self.training_data.getSequence(genomeIdx, contigIdx, 0)
                if frameIdx == 0:
                    rc = False
                    dnapos = rawpos
                else:
                    rc = True
                    dnapos = rawpos + sitelen - 1 # site starts at the reverse end

            pos = pc.rc_to_fwd(dnapos, len(sequence)) if rc else dnapos
            strand = '-' if rc else '+'
            occs.append(Links.Occurrence(sequence, pos, strand, sitelen, int(profileIdx)))

        return occs
    

    def frame_dimension_size(self) -> int:
        if self.training_data.datamode == DataMode.DNA:
            return 2 # fwd, rc
        else:
            return 6


    def N(self) -> int:
        return len(self.training_data.getTrainingData())
    

    def getDataset(self, 
                   repeat: bool = False,
                   withPosTracking: bool = False, 
                   original_data: bool = False):
        """ Returns a tensorflow dataset that yields batches of tiles from the given genomes. """
        genomes = self.training_data.getTrainingData(fromSource=original_data)

        second_out_sig = tf.TensorSpec(shape = (self.tiles_per_X, self.N(), self.frame_dimension_size(), 4), 
                                       dtype = tf.int32) if withPosTracking \
                            else tf.TensorSpec(shape = (0,), dtype = tf.int32)
        
        ds = tf.data.Dataset.from_generator(
            createBatch,
            args = (self.tiles_per_X, 
                    self.tile_size, 
                    self.alphabet, 
                    self.frame_dimension_size(),
                    genomes, 
                    self.training_data.reverse_frame_ids,
                    withPosTracking),
            output_signature = (tf.TensorSpec(shape = (self.tiles_per_X, self.N(), self.frame_dimension_size(), 
                                                       self.tile_size, self.alphabet_size()), 
                                              dtype = tf.float32),
                                second_out_sig)
        )
        
        if repeat:
            ds = ds.repeat()
        
        if self.batch_size is not None:
            ds = ds.batch(self.batch_size)
            
        if self.prefetch is not None:
            ds = ds.prefetch(self.prefetch)

        return ds
        
        
    def getRawData(self, fromSource: bool = False) -> list[list[list[str]]]:
        """ Returns a list of lists of lists of TileableSequence with the (possibly translated, depends on datamode) 
        training data. Outer list: genomes/species, second list: sequences, inner list: frames ([fwd, rc] for DNA data,
        6 frames for translated data). If fromSource is False (default), uses the internal copy. Otherwise, a new 
        extraction from the untouched source data is returned. """
        return self.training_data.getTrainingData(fromSource=fromSource)
    

    def softmask(self, genome_idx: int, sequence_idx: int, frame_idx: int, start_pos: int, masklen: int) -> str:
        """ Softmask the specified part of the training data, so that it is not used for further training. Uses the 
        internal copy and leaves the original data untouched. 
        Returns the specified part of the training data as string _before_ softmaskin (i.e. conv. to lower case). If the
        specified part is (partly) out of bounds, returns a shorter or an empty string. """
        trainingData = self.training_data.getTrainingData(fromSource=False)

        assert start_pos >= 0, f"[ERROR] >>> start_pos must be >= 0, not {start_pos}"
        assert masklen > 0, f"[ERROR] >>> masklen must be > 0, not {masklen}"
        assert genome_idx in range(len(trainingData)), \
            f"[ERROR] >>> No sequence with genome index {genome_idx}"
        assert sequence_idx in range(len(trainingData[genome_idx])), \
            f"[ERROR] >>> No sequence with inner index {sequence_idx} in genome {genome_idx}"
        assert frame_idx in range(len(trainingData[genome_idx][sequence_idx])), \
            f"[ERROR] >>> No sequence with frame index {frame_idx} in sequence {sequence_idx} of genome {genome_idx}"
        
        tseq = trainingData[genome_idx][sequence_idx][frame_idx]
        if start_pos >= len(tseq):
            logging.warning(f"[WARNING] >>> start_pos {start_pos} >= len(tseq) {len(tseq)}")
            return ""
        
        end_pos = start_pos + masklen
        if end_pos > len(tseq):
            logging.warning(f"[WARNING] >>> end_pos {end_pos} > len(tseq) {len(tseq)}")
        
        end_pos = min(end_pos, len(tseq))
        rseq = tseq[start_pos:end_pos] # return this
        softmasked = tseq[start_pos:end_pos].lower()
        # remove * if Translated and softmasked contains it
        if self.training_data.datamode == DataMode.Translated and '*' in softmasked:
            softmasked = softmasked.replace('*', ' ') if not self.training_data.replaceSpaceWithX \
                            else softmasked.replace('*', 'X')
            
        smseq = tseq[:start_pos] + softmasked + tseq[end_pos:]
        trainingData[genome_idx][sequence_idx][frame_idx] = smseq

        return rseq
    


def siteConversionHelper(mds: ModelDataSet, sitetuples: list[tuple], sitelen = 1) -> list[Links.Occurrence]:
    """ Wrapper around ModelDataSet.convertModelSites(). 
    Takes a list of tuples with sites that have to be defined by a genome index, contig index, frame and position. The
    tuples have to have at least 4 integer elements and the order of the elements has to be genome index, contig index,
    frame, position. The function then converts these tuples to a list of Links.Occurrence objects with the profileIdx
    set to -1. 

    Example:
        sitetuples = [(0, 0, 3, 40), (1, 0, 1, 50)]
        
        this will interpret the tuples to mean:
        - site 1: genome 0, contig 0, frame 3, position 40
        - site 2: genome 1, contig 0, frame 1, position 50
    """
    
    assert len(sitetuples) > 0, f"[ERROR] >>> No sites to convert"
    assert all([len(t) >= 4 for t in sitetuples]), \
        f"[ERROR] >>> Not all tuples have the required length of 4 or more"

    sites = np.zeros((len(sitetuples), 6), dtype=np.int32)
    for i, t in enumerate(sitetuples):
        # second dim: (genomeIdx, contigIdx, frameIdx, tileStartPos, tilePos, profileIdx)
        sites[i, 0] = t[0]
        sites[i, 1] = t[1]
        sites[i, 2] = t[2]
        sites[i, 3] = 0
        sites[i, 4] = t[3]
        sites[i, 5] = -1

    return mds.convertModelSites(sites, sitelen)