#!/usr/bin/env python
# coding: utf-8

# Obtaining Genome Sequences and Conversion to Tensor

# import libraries
import numpy as np

# import own modules
import sequtils as su



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
