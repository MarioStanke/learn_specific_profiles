#!/usr/bin/env python
# coding: utf-8

# Obtaining Genome Sequences and Conversion to Tensor

# import libraries
import numpy as np
import sys

# import own modules
sys.path.insert(0, 'MSAgen/')
import MSAgen
import SequenceRepresentation as sr
import sequtils as su



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
