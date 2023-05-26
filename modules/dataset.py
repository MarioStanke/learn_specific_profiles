import numpy as np
import tensorflow as tf

import MSAgen.MSAgen as MSAgen
import SequenceRepresentation as sr
import sequtils as su

def backGroundAAFreqs(genomes, verbose:bool = False):
    """
    Commpute vector of background frequencies of conceptually 
    translated amino acids, this is not the amino acid frequency
    in real protein sequences.
    
    genomes: list of N lists of nucleotide strings
    returns:
    vector Q of shape aa_alphabet_size
    """
    N = len(genomes)
    I = np.eye(su.aa_alphabet_size + 1)
    Q = np.zeros(su.aa_alphabet_size, dtype=np.float32)
    for i in range(N):
        for ctg in genomes[i]:
            aa_seqs = su.six_frame_translation(ctg)
            for frame in range(6):
                aa_seq = aa_seqs[frame]
                x = su.to_idx(aa_seq)
                Q += I[x].sum(axis=0)[1:] # skip missing AA (' ')
    sum = Q.sum()
    if sum > 0:
        Q /= Q.sum()
    if verbose:
        print ("background freqs: ", sum, "*")
        for c in range(su.aa_alphabet_size):
            print (f"{su.aa_alphabet[c+1]} {Q[c]:.4f}")
    return Q



def translateSequenceTiles(sequence, fwd_a, rc_b, tilesize):
    """	Get a tile of size `tilesize` from `sequence` and translate it to six aa-sequences, 
        starting at position `fwd_a` for the forward translation and 
        ending at position `rc_b` for the reverse complement translations. Returns a list of six strings and a boolean
        that indicates whether the tile covers the end of the sequence (true). """
    assert fwd_a >= 0, str(fwd_a)+" < 0"
    assert rc_b > 0, str(rc_b)+" <= 0"
    assert tilesize % 3 == 0, str(tilesize)+" % 3 not 0 ("+str(tilesize%3)+")"
    
    seqlen = len(sequence)
    assert fwd_a < seqlen, str(fwd_a)+", "+seqlen
    assert rc_b <= seqlen, str(rc_b)+", "+seqlen
    
    fwd_b = min(seqlen, fwd_a+tilesize+2) # add two extra positions for frames 1 and 2
    rc_a = max(0, rc_b-tilesize-2)
    
    fwd_seq = sequence[fwd_a:fwd_b]
    rc_seq = sequence[rc_a:rc_b]
    rc_seq = rc_seq[::-1].translate(su.rctbl)
    #print("DEBUG >>> Sequence:\n'"+sequence+"'")
    assert len(fwd_seq) == len(rc_seq), f"\nfwd: {fwd_a}: {fwd_b} -> {len(fwd_seq)}\n" \
                                         + f"rc: {rc_a}: {rc_b} -> {len(rc_seq)}\nseqlen: {seqlen}\n'{fwd_seq}'"
        
    aa_seqs = su.three_frame_translation(fwd_seq)
    aa_seqs.extend(su.three_frame_translation(rc_seq))
    if fwd_b == seqlen:
        assert rc_a == 0, str(rc_a)

    return aa_seqs, fwd_b == seqlen



def oneHot(aa_seq):
    """ One-hot encode an amino acid sequence into a numpy array of shape (len(aa_seq), aa_alphabet_size) """
    I = np.eye(su.aa_alphabet_size + 1) # for numpy-style one-hot encoding
    x = su.to_idx(aa_seq, su.aa_idx)
    one_hot = I[x] # here still aa_alphabet_size + 1 entries
    one_hot = one_hot[:,1:] 
    return one_hot



def restoreGenomePosition(aaTilePos, start, k, fwd: bool):
    """ From a tile starting at `start`, calculate the left position of a k-mer in the genome """
    p = aaTilePos*3 # to dna coord
    if fwd:
        p += start
    else:
        p = start - p - (k*3) + 1

    return p


        
rev_aa_idx = dict((i,c) for i,c in enumerate(su.aa_alphabet))
def to_aa(onehot, dropEmptyColumns = True):
    """ Translate a onehot encoded matrix to an amino acid sequence """
    assert onehot.shape[1] == 21, str(onehot.shape)
    aa_seq = ""
    for c in range(onehot.shape[0]):
        if dropEmptyColumns and np.max(onehot[c,:]) != 1:
            continue
        elif np.max(onehot[c,:]) != 1:
            aa_seq += ' '
        else:
            aa_idx = np.argmax(onehot[c,:])
            assert onehot[c, aa_idx] == 1, str(onehot[c,:])+", "+str(aa_idx)+", "+str(onehot[c,aa_idx])
            aa_idx += 1 # argmax + 1 as in translation, empty aa is cut out
            aa_seq += rev_aa_idx[aa_idx]
        
    return aa_seq



def seqlistFromGenomes(genomes: list[sr.Genome]) -> list[list[str]]:
    """ Create a list of lists of sequences from a list of SequenceRepresentation.Genomes """
    seqlist = []
    for g, genome in enumerate(genomes):
        seqlist.append([])
        for contig in genome:
            seqlist[g].append(contig.getSequence())

    return seqlist


# Use a generator to get genome batches, simplified position handling
def createBatch(ntiles: int, aa_tile_size: int, genomes: list[list[str]], withPosTracking: bool = False):
    """ Generator function to create batches of tiles from a list of genomes. 
        Returns a tuple of (X, Y) where X is a numpy array of shape (ntiles, N, 6, aa_tile_size, 21) that contains
          the one-hot encoded tiles and Y is a numpy array of shape (ntiles, N, 6, 4) if `withPosTracking` was true
          (otherwise an empty list) that contains the position information of the tiles in the genome. Y[a,b,c,:] is a
          list of [genomeID, contigID, tile_start, tile_length] for the tile `a` in genome `b` at fram `c` in X.
          Set `withPosTracking` to true to be able to restore the position of a k-mer in the genome from the tile. """
    
    assert aa_tile_size >= 1, "aa_tile_size must be positive, non-zero (is: "+str(aa_tile_size)+")"
    tile_size = aa_tile_size * 3 # tile_size % 3 always 0
    N = len(genomes)
    state = [{'idx': 0, 'fwd_a': 0, 'rc_b': None, 'exhausted': (len(seqs) == 0)} for seqs in genomes]

    while not all(s['exhausted'] for s in state):
        X = np.zeros([ntiles, N, 6, aa_tile_size, su.aa_alphabet_size], dtype=np.float32)   # (tilesPerX, N, 6, T, 21)
        I = np.eye(su.aa_alphabet_size + 1) # for numpy-style one-hot encoding
        if withPosTracking:
            # [:,:,:,0] - genome IDs, 
            # [:,:,:,1] - contig IDs, 
            # [:,:,:,2] - start pos of X[:,:,:,0,:] w.r.t. genome sequence, 
            # [:,:,:,3] - aa seqlen in that tile and frame, -1: exhausted
            # fwd/rc start both refer to the 0-based position of the 0th position in the respective tile 
            #   _on the fwd strand_
            posTrack = np.ones([ntiles, N, 6, 4], dtype=np.int32) *-1

        for t in range(ntiles):
            for g in range(N):
                if state[g]['exhausted']:
                    continue
                    
                sidx = state[g]['idx']
                slen = len(genomes[g][sidx])
                if state[g]['rc_b'] is None:
                    state[g]['rc_b'] = slen

                fwd_a = state[g]['fwd_a']
                fwd_b = min(slen, fwd_a+tile_size) # seq[fwd_a:fwd_b]   -> a>>>>>>>b
                rc_b = state[g]['rc_b']
                rc_a = max(0, rc_b-tile_size)      # rc(seq[rc_a:rc_b]) -> b<<<<<<<a
                if withPosTracking:
                    posTrack[t,g,:,0] = g
                    posTrack[t,g,:,1] = sidx
                    posTrack[t,g,0,2] = fwd_a
                    posTrack[t,g,1,2] = fwd_a+1
                    posTrack[t,g,2,2] = fwd_a+2
                    posTrack[t,g,3,2] = rc_b-1
                    posTrack[t,g,4,2] = rc_b-2
                    posTrack[t,g,5,2] = rc_b-3
                    
                assert sidx < len(genomes[g]), str(sidx)+" >= "+str(len(genomes[g]))+" for genome "+str(g)
                
                # translate and add tiles
                sequence = genomes[g][sidx]
                if type(sequence) is not str:
                    # with tf, input are byte-strings and need to be converted back
                    sequence = tf.compat.as_str(sequence) 
                
                aa_seqs, seqExhausted = translateSequenceTiles(sequence, fwd_a, rc_b, tile_size)
                for frame in range(6):
                    aa_seq = aa_seqs[frame]
                    assert len(aa_seq) <= aa_tile_size, f"{len(aa_seq)} != {aa_tile_size}, " \
                                          +f"fwd_a, rc_b, slen, tile, genome, frame: {(fwd_a, rc_b, slen, t, g, frame)}"
                    x = su.to_idx(aa_seq, su.aa_idx)
                    num_aa = x.shape[0]
                    if withPosTracking:
                        posTrack[t,g,frame,3] = num_aa
                        
                    if (num_aa > 0):
                        one_hot = I[x] # here still aa_alphabet_size + 1 entries
                        # missing sequence will be represented by an all-zero vector
                        one_hot = one_hot[:,1:] 
                        X[t,g,frame,0:num_aa,:] = one_hot
                        
                # update state
                state[g]['fwd_a'] = fwd_b
                state[g]['rc_b'] = rc_a
                if seqExhausted:
                    state[g]['idx'] += 1
                    state[g]['fwd_a'] = 0
                    state[g]['rc_b'] = None
                    
                if state[g]['idx'] == len(genomes[g]):
                    state[g]['exhausted'] = True
        
        if withPosTracking:
            yield X, posTrack
        else:
            yield X, []



def getDataset(genomes: list[list[str]], tiles_per_X: int, tile_size: int, withPosTracking: bool = False):
    """ Returns a tensorflow dataset that yields batches of tiles from the given genomes. """
    if withPosTracking:
        # use deprecated way if tensorflow version is < 2.4
        if tf.__version__.split('.')[0:2] < ['2','4']:
            ds = tf.data.Dataset.from_generator(
                createBatch,
                args = (tf.constant(tiles_per_X), tf.constant(tile_size), tf.constant(genomes, dtype=tf.string), 
                        tf.constant(True)),
                output_types = (tf.float32, tf.int32),
                output_shapes = (tf.TensorShape([tiles_per_X, len(genomes), 6, tile_size, su.aa_alphabet_size]),
                                 tf.TensorShape([tiles_per_X, len(genomes), 6, 4]))
            )
        else:
            ds = tf.data.Dataset.from_generator(
                createBatch,
                args = (tf.constant(tiles_per_X), tf.constant(tile_size), tf.constant(genomes, dtype=tf.string), 
                        tf.constant(True)),
                output_signature = (tf.TensorSpec(shape = (tiles_per_X, len(genomes), 6, tile_size, 
                                                            su.aa_alphabet_size), 
                                                  dtype = tf.float32),
                                    tf.TensorSpec(shape = (tiles_per_X, len(genomes), 6, 4),
                                                  dtype = tf.int32))
            )
    else:
        if tf.__version__.split('.')[0:2] < ['2','4']:
            ds = tf.data.Dataset.from_generator(
                createBatch,
                args = (tf.constant(tiles_per_X), tf.constant(tile_size), tf.constant(genomes, dtype=tf.string), 
                        tf.constant(False)),
                output_types = (tf.float32, tf.float32),
                output_shapes = (tf.TensorShape([tiles_per_X, len(genomes), 6, tile_size, su.aa_alphabet_size]),
                                 tf.TensorShape(0))
            )
        else:
            ds = tf.data.Dataset.from_generator(
                createBatch,
                args = (tf.constant(tiles_per_X), tf.constant(tile_size), tf.constant(genomes, dtype=tf.string), 
                        tf.constant(False)),
                output_signature = (tf.TensorSpec(shape = (tiles_per_X, len(genomes), 6, tile_size, 
                                                            su.aa_alphabet_size),
                                                  dtype = tf.float32),
                                    tf.TensorSpec(shape = (0),
                                                  dtype = tf.float32))
            )
    return ds



class DatasetHelper:
    """ Helper class to easily pass dataset parameters around and tweak dataset creation as needed """
    
    def __init__(self, 
                 genomes,
                 tiles_per_X: int,
                 tile_size: int,
                 batch_size: int = None,
                 prefetch: int = None):
        """ If batch_size and/or prefetch are None (default), 
            no batch size and/or prefetch dataset is created unless
            specified in the call to DatasetHelper.getDataset() """
        self.genomes = genomes
        self.tilesPerX = tiles_per_X
        self.tileSize = tile_size
        self.batchSize = batch_size
        self.prefetch = prefetch
        
    def getDataset(self, 
                   tiles_per_X: int = None,
                   tile_size: int = None,
                   batch_size: int = None,
                   prefetch: int = None,
                   repeat: bool = False,
                   withPosTracking: bool = False):
        """ Any argument specified here overwrites the defaults from object constuction """
        tilesPerX = tiles_per_X if tiles_per_X is not None else self.tilesPerX
        tileSize = tile_size if tile_size is not None else self.tileSize
        batchSize = batch_size if batch_size is not None else self.batchSize
        prefetch_ = prefetch if prefetch is not None else self.prefetch
        ds = getDataset(self.genomes,
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

    def allUC(self):
        """ Make all bases in the genome upper case (in place!, useful after training with reporting) """
        for g in range(len(self.genomes)):
            for c in range(len(self.genomes[g])):
                self.genomes[g][c] = self.genomes[g][c].upper()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Simulate data

# create a set of positions at which a new k-mer should not be inserted to avoid overlap
def getForbiddenPositions(pos:list, k: int, slen: int):
    fp = set()
    for p in pos:
        fp.update(list(range(max(0, p-k+1), min(p+k, slen)))) # insert of k-mer at p-k+1 would lead to overlap, so avoid
        
    return fp



def insertPatternsToGenomes(patterns: list[str], 
                            genomes: list[sr.Genome],
                            dna_alphabet: str,
                            mutationProb = 0, 
                            multiplyPattern: range = None,
                            repeatPattern: range = None,
                            isRepeat: bool = False,
                            forbiddenPositions = None, # dict {Sequence.id: set of forbidden positions}
                            verbose = False):
    """
        Parameters:
            patterns            list of nucleotide strings used as insert elements
            genomes             list of SequenceRepresentation.Genome objects
            dna_alphabet        string of nucleotide characters
            mutationProb        probability of mutation of inserted pattern at an average site
            multiplyPattern     range of integers, sample from this range to concatenate each pattern multiple times.
                                    If None, do not multiply but keep single pattern
            repeatPattern       range of integers, sample from this range to insert each pattern multiple times into 
                                    each genome. If None, do not repeat but insert each pattern only once per genome
            isRepeat            if True, insert patterns as repeats, otherwise insert as single copy (has influence on 
                                    mutationProb)
            forbiddenPositions  dict of sets of positions that are not allowed for insertions
            verbose             print information about inserted patterns
    """	

    if multiplyPattern is not None:
        assert type(multiplyPattern) == range or type(multiplyPattern) == list, \
            f"[ERROR] >>> multiplyPattern must be a range or a list of integers, not {type(multiplyPattern)}"
        assert sorted(list(multiplyPattern))[0] > 0, \
            "[ERROR] >>> multiplyPattern must be a range or list of integers > 0"
    if repeatPattern is not None:
        assert type(repeatPattern) == range or type(repeatPattern) == list, \
            f"[ERROR] >>> repeatPattern must be a range or a list of integers, not {type(repeatPattern)}"
        assert sorted(list(repeatPattern))[0] > 0, "[ERROR] >>> repeatPattern must be a range or list of integers > 0"

    if forbiddenPositions is None:
        forbiddenPositions = {}
    
    dna_alphabet_size = len(dna_alphabet)
    for pattern in patterns:
        if verbose: # print translated peptide
            print (f"Pattern {pattern} translates to ", su.six_frame_translation(pattern))
            
        # if multiple, insert multiple copies of pattern
        pattern = pattern*np.random.choice(multiplyPattern) if multiplyPattern is not None else pattern
        assert pattern != '', "[ERROR] >>> Pattern is empty, use at least range(1,2) for multiplyPattern!"
        plen = len(pattern)
        for genome in genomes:
            # mutate pattern
            mutatedPattern = ""
            for (r, base) in enumerate(pattern):
                if isRepeat:
                    mutate = np.random.binomial(1, mutationProb)
                else:
                    mutate = r%3==2 and np.random.binomial(1, 3*mutationProb) # only mutate at 3rd position of codon

                mutate = np.random.binomial(1, mutationProb) # binary: mutate or not
                if mutate: # mutate
                    # the mutated character is not allowed to be the same as base
                    c = np.random.choice(dna_alphabet_size - 1)
                    d = (su.nuc_idx[base] + c ) % dna_alphabet_size
                    b = dna_alphabet[d]
                    mutatedPattern += b
                else: # keep the original character
                    mutatedPattern += base
                    
            assert len(mutatedPattern) == plen
                    
            # insert mutated pattern in random place in genome under uniform distribution
            seqsizes = [len(sequence) for sequence in genome]
            relsizes = seqsizes / np.sum(seqsizes)
            ninserts = np.random.choice(repeatPattern) if repeatPattern is not None else 1
            assert ninserts > 0, f"[ERROR] >>> ninserts must be > 0, check repeatPattern argument ({repeatPattern})!"
            for _ in range(ninserts):
                # chose a random contig j proportional to its size
                j = np.random.choice(a=range(len(genome)), p=relsizes)
                contig = genome[j]
                assert contig.sequence is not None, f"[ERROR] >>> Contig {contig} has no sequence!"
                if contig.id not in forbiddenPositions:
                    forbiddenPositions[contig.id] = set()

                # next, choose a random position in that contig
                assert len(forbiddenPositions[contig.id]) < len(contig), \
                    f"[ERROR] >>> Too many inserts for contig {contig}"
                foundPos = False
                while not foundPos:
                    pos = np.random.choice(len(contig) - plen)
                    foundPos = pos not in forbiddenPositions[contig.id]
                    
                # make sure inserts do not overlap
                forbiddenPositions[contig.id].update(getForbiddenPositions([pos], plen, len(contig)))
                
                # replace the string starting at pos in contig with mutatedPattern
                contig.sequence = contig.sequence[0:pos] + mutatedPattern + contig.sequence[pos+plen:]
                inserttype = "repeat" if isRepeat else "pattern"
                contig.addSubsequenceAsElement(pos, pos+plen, inserttype, genomic_positions = True, no_elements = True)
                if verbose:
                    print (f"  mutated to {mutatedPattern} and inserted in contig {contig} at position {pos}")

    return forbiddenPositions



def getRandomGenomes(N: int, genome_sizes: list[int],
                     insertPatterns: list[str] = None,
                     repeatPatterns: list[str] = None, 
                     mutationProb = 0,
                     multiplyRepeat: range = range(2,10),
                     multipleRepeatInserts: range = range(5,10),
                     verbose = False) -> list[sr.Genome]:
    """ 
    Construct random genomes with inserted patterns and/or repeats

      Parameters:
        N                      number of genomes
        genome_sizes           list of N lists of sizes in nucleotides
        insertPatterns         list of nucleotide strings used as insert elements
        repeatPatterns         list of nucleotide strings used as repeat elements
        mutationProb           probability of mutation of inserted pattern at an average site
        multiplyRepeat         range of integers, sample from this range to concatenate multiple repeatPatterns copies
        multipleRepeatInserts  range of integers, sample from this range to insert multiple repeatPatterns per genome
      
      Returns:
        genomes  list of N SequenceRepresentation.Genome objects possibly with inserted patterns and/or repeats
    """
    # construct random genome sequences
    basic_dna_alphabet = "ACGT"
    genomes = [sr.Genome() for _ in range(N)]
    for i, genome in enumerate(genomes):
        for j, ctglen in enumerate(genome_sizes[i]):
            genome.addSequence(sr.Sequence(f"genome_{i}", f"chr_{j}", "+", 0, 
                                           sequence = ''.join(np.random.choice(list(basic_dna_alphabet), ctglen))))
    
    forbiddenPositions = {}
    if repeatPatterns is not None:
        forbiddenPositions = insertPatternsToGenomes(patterns=repeatPatterns, 
                                                     genomes=genomes, 
                                                     dna_alphabet=basic_dna_alphabet, 
                                                     mutationProb=mutationProb*10,
                                                     multiplyPattern=multiplyRepeat,
                                                     repeatPattern=multipleRepeatInserts,
                                                     isRepeat=True,
                                                     verbose=verbose)
    
    # insert relevant patterns, make sure no repeats are overwritten
    if insertPatterns is not None:
        insertPatternsToGenomes(patterns=insertPatterns, 
                                genomes=genomes, 
                                dna_alphabet=basic_dna_alphabet, 
                                mutationProb=mutationProb, 
                                isRepeat=False, 
                                forbiddenPositions=forbiddenPositions,
                                verbose = verbose)

    return genomes



def simulateGenomes(N, seqlen, genelen,
                    cdist = 0.05, ncdist = 0.1, tree = 'star', omega = 0.4) -> list[sr.Genome]:
    """ Use MSAgen to generate data. 

        Parameters:
            N       number of single-sequence genomes to generate
            seqlen  total lenght of each generated sequence
            genelen length of the simulated ortholog gene inside the sequences, rounded up to a multiple of 3
            cdist   height of the underlying tree for coding sequences
            ncdist  height of the underlying tree for non-coding sequences
            tree    type of tree, either "star" or "caterpillar"
            omega   codon regions under negative selection
    """
    
    sequences, posDict = MSAgen.generate_sequences(N, seqlen, genelen, 
                                                    coding_dist=cdist, noncoding_dist=ncdist, tree=tree, omega=omega)
    
    genomes = [sr.Genome([sr.Sequence(f"genome_{i}", seq.id, "+", 0, sequence = str(seq.seq))]) \
                   for i, seq in enumerate(sequences)]
    for genome in genomes:
        sequence = genome[0]
        if posDict['5flank_start'] is not None:
            assert posDict['5flank_len'] is not None, \
                f"[ERROR] >>> 5flank_len must be set if 5flank_start is set! {posDict}"
            sequence.addSubsequenceAsElement(posDict['5flank_start'], posDict['5flank_start']+posDict['5flank_len'],
                                             "5flank", genomic_positions = True, no_elements = True)
            
        if posDict['start_codon'] is not None:
            sequence.addSubsequenceAsElement(posDict['start_codon'], posDict['start_codon']+3,
                                             "start_codon", genomic_positions = True, no_elements = True)
        
        if posDict['cds_start'] is not None:
            assert posDict['cds_len'] is not None, \
                f"[ERROR] >>> cds_len must be set if cds_start is set! {posDict}"
            sequence.addSubsequenceAsElement(posDict['cds_start'], posDict['cds_start']+posDict['cds_len'],
                                             "cds", genomic_positions = True, no_elements = True)
            
        if posDict['stop_codon'] is not None:
            sequence.addSubsequenceAsElement(posDict['stop_codon'], posDict['stop_codon']+3,
                                             "stop_codon", genomic_positions = True, no_elements = True)
            
        if posDict['3flank_start'] is not None:
            assert posDict['3flank_len'] is not None, \
                f"[ERROR] >>> 3flank_len must be set if 3flank_start is set! {posDict}"
            sequence.addSubsequenceAsElement(posDict['3flank_start'], posDict['3flank_start']+posDict['3flank_len'],
                                             "3flank", genomic_positions = True, no_elements = True)
        
    return genomes