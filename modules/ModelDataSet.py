from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
import tensorflow as tf

from . import dataset
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



def backgroundFreqs(sequences: list[list[list[TileableSequence]]], alphabet: list[str], verbose: bool = False):
    """
    Commpute vector of background frequencies of letters in the sequences. Letters not in the alphabet are ignored.
    
        sequences: list of N lists of strings
    Returns:
        vector Q of shape len(alphabet)
    """
    Q = np.zeros(alphabet.size(), dtype=np.float32)
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
                genomes: list[list[list[TileableSequence]]], 
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

                if all([s.finished for s in sequences]):
                    state[g]['sequence_idx'] += 1
                    if state[g]['sequence_idx'] == len(genomes[g]):
                        state[g]['exhausted'] = True
                        state[g]['sequence_idx'] = None
                        
        if withPosTracking:
            yield X, posTrack
        else:
            yield X, []



class DataMode(Enum):
    DNA = 1
    Translated = 2


class _TrainingDataWrapper:
    """ Takes care of providing the training data for a model. Depending on the datamode, the data is either used as
    genomic sequences or translated sequences. The data is stored as a list of Genomes and can be accessed as a list of
    list of list of strings (genomes[sequences[frames]]). The latter is used for training. The class also provides a 
    mapping of the training sequences to the sr.Sequence or sr.TranslatedSequence objects. """
    def __init__(self, data: list[sr.Genome], datamode: DataMode, tile_size: int):
        assert datamode in DataMode, f"[ERROR] >>> datamode must be of type DataMode, not {type(datamode)}"
        self.datamode = datamode
        self._data = data
        self.tile_size = tile_size
        trainingseqs, mapping = self._extract_training_sequences()
        self._training_sequences = trainingseqs
        self._sequence_mapping = mapping


    def _extract_training_sequences(self) -> tuple[list[list[list[TileableSequence]]], \
                                                   dict[int, dict[int, dict[int, sr.Sequence|sr.TranslatedSequence]]]]:
        """ Takes the original data and extracts the sequences as a list of lists of lists of TileableSequence objects.
         Depending on datamode, either uses the original genomic sequences and adds the respective reverse complements,
         or translates the sequences in 6 frames. Returns as second element a mapping of nested list indices to the 
         original genomic Sequence or TranslatedSequence objects in the data."""
        if self.datamode == DataMode.DNA:
            return (
                [[[TileableSequence(s.getSequence(), self.tile_size, reverse_tiling=False), 
                   TileableSequence(s.getSequence(rc=True), self.tile_size, reverse_tiling=True)] for s in g] \
                    for g in self._data], 
                {i: {j: {0: s} for j, s in enumerate(g)} for i, g in enumerate(self._data)}
            )
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
                        tseq = sr.TranslatedSequence(s, frame)
                        sequence_mapping[i][j][frame] = tseq
                        seqframes.append(TileableSequence(tseq.getSequence(), self.tile_size, 
                                                          reverse_tiling = frame >= 3))

                    sequences.append(seqframes)

                training_sequences.append(sequences)

            return training_sequences, sequence_mapping
        

    def getTrainingData(self, fromSource: bool = False) -> list[list[list[TileableSequence]]]:
        """ Returns a list of lists of lists of TileableSequence with the (possibly translated, depends on datamode) 
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
# TODO: store translated sequences (or original sequence reference) as _presenting_data (or so) member, create dataset from that, edit posTracking accordingly

    def __init__(self, data: list[sr.Genome], datamode: DataMode, Q: np.ndarray = None,
                 tile_size: int = 334, tiles_per_X: int = 13, batch_size: int = 1, prefetch: int = 3):
        """
        Attributes:
            data: list of Genomes
            datamode: DataMode
            Q: np.ndarray -- Q-vector of background frequencies for the data, must be of same length as alphabet.size().
                             If None, it is calculated from the data
            tile_size: int -- size of the tiles during training
            tiles_per_X: int -- number of tiles per X
            batch_size: int -- number of batches to be used during training
            prefetch: int -- number of batches to prefetch
        """
        assert datamode in DataMode, f"[ERROR] >>> datamode must be of type DataMode, not {type(datamode)}"
        assert isinstance(data, list), f"[ERROR] >>> data must be of type list, not {type(data)}"
        assert len(data) > 1, f"[ERROR] >>> Need at least 2 genomes for training, not {len(data)}"
        assert all(typecheck(x, "Genome") for x in data), f"[ERROR] >>> data must be a list of Genomes"

        self.training_data = _TrainingDataWrapper(data, datamode, tile_size)
        self.alphabet = _DNA_ALPHABET if datamode == DataMode.DNA else _TRANSLATED_ALPHABET
        if Q is None:
            self.Q = backgroundFreqs(self.training_data.getTrainingData(), self.alphabet)

        assert self.Q.shape == (self.alphabet_size()), \
            f"[ERROR] >>> Q-matrix must have shape ({self.alphabet_size()},), not {self.Q.shape}"
        
        self.tile_size = tile_size
        self.tiles_per_X = tiles_per_X
        self.batch_size = batch_size
        self.prefetch = prefetch
        

    def alphabet_size(self) -> int:
        return len(self.alphabet)
    

    def frame_dimension_size(self) -> int:
        if self.training_data.datamode == DataMode.Original:
            return 2 # fwd, rc
        else:
            return 6


    def N(self) -> int:
        return len(self.training_data.getTrainingData())
    

    def getDataset(self, withPosTracking: bool = False, original_data: bool = False):
        """ Returns a tensorflow dataset that yields batches of tiles from the given genomes. """
        genomes = self.training_data.getTrainingData(fromSource=original_data)

        second_out_sig = tf.TensorSpec(shape = (self.tiles_per_X, self.N(), self.frame_dimension_size(), 4), 
                                       dtype = tf.int32) if withPosTracking \
                            else tf.TensorSpec(shape = (0), dtype = tf.float32)
        
        ds = tf.data.Dataset.from_generator(
            createBatch,
            args = (self.tiles_per_X, self.tile_size, self.alphabet, self.frame_dimension_size(),
                    genomes, withPosTracking),
            output_signature = (tf.TensorSpec(shape = (self.tiles_per_X, self.N(), self.frame_dimension_size(), 
                                                       self.tile_size, self.alphabet_size()), 
                                              dtype = tf.float32),
                                second_out_sig)
        )
        
        return ds
        
        