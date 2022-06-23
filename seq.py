#!/usr/bin/env python
# coding: utf-8

# Obtaining Genome Sequences and Conversion to Tensor

# import libraries
import numpy as np

# import own modules
import sequtils as su

# create a set of positions at which a new k-mer should not be inserted to avoid overlap
def getForbiddenPositions(pos:list, k: int, slen: int):
    fp = set()
    for p in pos:
        fp.update(list(range(max(0, p-k+1), min(p+k-1, slen))))
        
    return fp

def insertPatternsToGenomes(patterns:list, genomes,
                            dna_alphabet,
                            mutationProb = 0, 
                            repeat = False, 
                            repeatMultiple = range(2,10), # if repeat, sample from this range to concatenate multiple repeat copies
                            repeatInsert = range(5,10),   # if repeat, sample from this range to insert multiple repeats
                            forbiddenPositions = None,    # dict of dict of sets of positions at where no insert may start
                            verbose = False):
    # dict of dicts of dict, for each genome, map each contig to a dict that collects insert positions and patterns
    tracking = dict([(i, dict([(j, {'pos': [], 'pattern': []}) for j in range(len(genomes[i]))])) for i in range(len(genomes))])
    if forbiddenPositions is None:
        forbiddenPositions = dict([(i, dict([(j, set()) for j in range(len(genomes[i]))])) for i in range(len(genomes))])
    
    dna_alphabet_size = len(dna_alphabet)
    for pattern in patterns:
        if verbose: # print translated peptide
            print (f"Pattern {pattern} translates to ", su.six_frame_translation(pattern))
            
        # if multiple, insert multiple copies of pattern
        pattern = pattern*np.random.choice(repeatMultiple) if repeat else pattern
        assert pattern != '', "Pattern is empty, use at least range(1,2) for repeatMultiple!"
        plen = len(pattern)
        for i in range(len(genomes)):
            # mutate pattern
            mutatedPattern = ""
            for (r,a) in enumerate(pattern):
                if repeat:
                    mutate = np.random.binomial(1, mutationProb)
                else:
                    mutate = r%3==2 and np.random.binomial(1, 3*mutationProb)
                    
                if mutate: # mutate
                    # the mutated character is not allowed to be the same as a
                    c = np.random.choice(dna_alphabet_size - 1)
                    d = (su.nuc_idx[a] + c ) % dna_alphabet_size
                    b = dna_alphabet[d]
                    mutatedPattern += b
                else: # keep the original character
                    mutatedPattern += a
                    
            assert len(mutatedPattern) == plen
                    
            # insert mutated pattern in random place in genome under uniform distribution
            seqsizes = [len(genomes[i][j]) for j in range(len(genomes[i]))]
            relsizes = seqsizes / np.sum(seqsizes)
            ninserts = np.random.choice(repeatInsert) if repeat else 1
            for _ in range(ninserts):
                # chose a random contig j proportional to its size
                j = np.random.choice(a=range(len(genomes[i])), p=relsizes)
                # next, choose a random position in that contig
                assert len(forbiddenPositions[i][j]) < len(genomes[i][j]), "Too many inserts for genome "+str(i)+" seq "+str(j)
                foundPos = False
                while not foundPos:
                    pos = np.random.choice(len(genomes[i][j]) - plen)
                    foundPos = pos not in forbiddenPositions[i][j]
                    
                forbiddenPositions[i][j].update(getForbiddenPositions([pos], plen, len(genomes[i][j]))) # make sure inserts do not overlap
                
                # replace the string starting at pos in genomes[i][j] with mutatedPattern
                s = genomes[i][j]
                genomes[i][j] = s[0:pos] + mutatedPattern + s[pos+plen:]
                tracking[i][j]['pos'].append(pos)
                tracking[i][j]['pattern'].append(mutatedPattern)
                if verbose:
                    print (f"  mutated to {mutatedPattern} and inserted in genome {i}" +
                           f" contig {j} at position {pos}")
                    
    return tracking

def getRandomGenomes(N, genome_sizes,
                     insertPatterns:list = None,
                     repeatPatterns:list = None, 
                     mutationProb = 0,
                     repeatMultiple = range(2,10), # if repeat, sample from this range to concatenate multiple repeat copies
                     repeatInsert = range(5,10),   # if repeat, sample from this range to insert multiple repeats
                     verbose = False):
    """ 
      Parameters:
        N              number of genomes
        genome_sizes   list of N lists of sizes in nucleotides
        insertPatterns list of nucleotide strings
        repeatPatterns list of nucleotide strings
        mutationProb   probability of mutation of inserted pattern at an average site
      
      Returns:
        genomes        list of N nucleotide strings
    """
    # construct random genome sequences
    basic_dna_alphabet = "ACGT"
    genomes = [[''.join(np.random.choice(list(basic_dna_alphabet), ctglen))
           for ctglen in genome_sizes[i]]
           for i in range(N)]
    
    repeatTracking = None
    if repeatPatterns is not None:
        repeatTracking = insertPatternsToGenomes(patterns=repeatPatterns, 
                                                 genomes=genomes, 
                                                 dna_alphabet=basic_dna_alphabet, 
                                                 mutationProb=mutationProb*10, 
                                                 repeat=True,
                                                 repeatMultiple=repeatMultiple,
                                                 repeatInsert=repeatInsert,
                                                 verbose=verbose)

    # insert relevant patterns, make sure no repeats are overwritten
    insertTracking = None
    forbiddenPositions = None
    if repeatTracking is not None:
        forbiddenPositions = dict()
        for i in repeatTracking:
            forbiddenPositions[i] = dict()
            for j in repeatTracking[i]:
                forbiddenPositions[i][j] = set()
                assert all([len(repeatTracking[i][j]['pattern'][l]) == len(repeatTracking[i][j]['pattern'][0]) for l in range(len(repeatTracking[i][j]['pattern']))])
                forbiddenPositions[i][j].update(getForbiddenPositions(repeatTracking[i][j]['pos'],
                                                                      len(repeatTracking[i][j]['pattern'][0]),
                                                                      len(genomes[i][j])))
    if insertPatterns is not None:
        insertTracking = insertPatternsToGenomes(patterns=insertPatterns, 
                                                 genomes=genomes, 
                                                 dna_alphabet=basic_dna_alphabet, 
                                                 mutationProb=mutationProb, 
                                                 repeat=False, 
                                                 forbiddenPositions=forbiddenPositions,
                                                 verbose = verbose)

        #for pattern in insertPatterns:
        #    plen = len(pattern)
        #    if verbose: # print translated peptide
        #        print (f"Pattern {pattern} translates to ", su.six_frame_translation(pattern))
        #    for i in range(N):
        #        # mutate pattern
        #        mutatedPattern = ""
        #        for (r,a) in enumerate(pattern):
        #            if r%3==2 and np.random.binomial(1, 3*mutationProb): # mutate
        #                # the mutated character is not allowed to be the same as a
        #                c = np.random.choice(su.dna_alphabet_size - 1)
        #                d = (su.nuc_idx[a] + c ) % su.dna_alphabet_size
        #                b = su.dna_alphabet[d]
        #                mutatedPattern += b
        #            else: # keep the original character
        #                mutatedPattern += a
        #        # insert mutated pattern in random place in genome under uniform distribution
        #        relsizes = genome_sizes[i] / np.sum(genome_sizes[i])
        #        # chose a random contig j proportional to its size
        #        j = np.random.choice(a=range(len(genome_sizes[i])), p=relsizes)
        #        # next, choose a random position in that contig
        #        pos = np.random.choice(genome_sizes[i][j] - plen)
        #        # replace the string starting at pos in genomes[i][j] with mutatedPattern
        #        s = genomes[i][j]
        #        genomes[i][j] = s[0:pos] + mutatedPattern + s[pos+plen:]
        #        if verbose:
        #            print (f"  mutated to {mutatedPattern} and inserted in genome {i}" +
        #                  f" contig {j} at position {pos}")
    return genomes, repeatTracking, insertTracking

# ### Convert genomes to tensor
# Let $$X \in \{0,1\}^{B\times N\times 6 \times T \times 21}$$ be a batch of **one-hot-like encoded input translated sequences**,
# where $B$ is `batch_size`, $N$ is the number of genomes and $T$ is the `tile_size` (in aa).
# The 6 is here the number of translated frames in order (0,+),(1,+),(2,+),(0,-),(1,-),(2,-).
# The 21 is here the size of the considered amino acid alphabet.


def getNextBatch(genomes, batch_size, tile_size, verbose:bool = False):
    """
    Convert next batch of sequence tiles to a tensor X.
    This should be called once for a set of genomes.
    The converted tensors are either held in memory or
    serialized to disc for quick repeated multiple access during training.
    
    genomes: list of N lists of nucleotide strings
             genomes is consumed (changed) so that iterated calls of getNextBatch eventually result
             in empty lists. Genome sequences itself should not be empty strings.
    returns:
    tensor of shape
    """
    N = len(genomes)
    # test whether any sequenes are left
    i=0
    while i<N and not genomes[i]:
        i += 1
    if i == N: # all lists empty, genomes is exhausted
        print("returning none")
        return None
    
    X = np.zeros([batch_size, N, 6, tile_size, su.aa_alphabet_size], dtype=np.float32)
    I = np.eye(su.aa_alphabet_size + 1) # for numpy-style one-hot encoding
    for b in range(batch_size):
        for i in range(N):
            # get next up to tile_size amino acids from genome i
            slen = 0
            if not genomes[i]:
                continue # i-th genome already exhausted
                
            slen = len(genomes[i][0])
            translatable_seq = genomes[i][0][:min(slen,
                                                  3 * tile_size + 2)]

            # some nucleotides are part of both neighboring tiles
            aa_seqs = su.six_frame_translation(translatable_seq)
            for frame in range(6):
                aa_seq = aa_seqs[frame]
                x = su.to_idx(aa_seq)
                num_aa = x.shape[0]
                if (num_aa > 0):
                    one_hot = I[x] # here still aa_alphabet_size + 1 entries
                    # missing sequence will be represented by an all-zero vector
                    one_hot = one_hot[:,1:] 
                    X[b,i,frame,0:num_aa,:] = one_hot
                if verbose:
                    print (f"b={b} i={i} f={frame} len={len(aa_seq):>2} {aa_seq:<{tile_size}}")
                    # print(x, X[b,i,frame])

            # remove from genome sequence, what has been used
            if len(genomes[i][0]) > 3 * tile_size:
                genomes[i][0] = genomes[i][0][3 * tile_size : ]
            else: # the rest of the sequence has been used
                genomes[i].pop(0)
                
    print("returning X")
    return X


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
