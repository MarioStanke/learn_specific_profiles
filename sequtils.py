# Utility functions for DNA and amino acid sequences
# Mario Stanke, August 2021
import numpy as np
import pandas as pd

dna_alphabet = "ACGTacgtNWSMKRYBDHVNZ" # for real sequences, need to know about softmask and ambiguous bases
complements =  "TGCAtgcaNNNNNNNNNNNNN" # also map softmask, map ambiguous to N
rctbl = str.maketrans(dna_alphabet, complements)
dna_alphabet_size = len(dna_alphabet)

codon_len = 3
codon_alphabet_size = dna_alphabet_size ** codon_len # 64
genetic_code = { # translation table 1 of NCBI
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W', 
}

aa_alphabet =[' ', # missing value, 0 used for padding
              'C', 'K', 'E', 'W', 'T', 'G', 'Y', 'A', 'I', 'N', # 20 regular
              'V', 'H', 'S', 'D', 'F', 'M', 'R', 'L', 'P', 'Q', # amino acids
             '*' # stop codon
             ]
aa_alphabet_size = len(aa_alphabet) - 1 # do not count ' '

def six_frame_translation(S):
    """ return all 6 conceptually translated protein sequences """
    T = []
    for seq in (S, S[::-1].translate(rctbl)): # forward, reverse-complement sequence
        for f in range(3): # frame
            prot = ""
            for i in range(f, len(S) - codon_len + 1, codon_len):
                codon = seq[i:i+codon_len]
                if codon not in genetic_code: # real sequences may contain N's or softmasking or ambiguous bases
                    prot += ' '                    # use null aa in that case
                else:
                    prot += genetic_code[codon]
                #prot += genetic_code[seq[i:i+codon_len]]
            T.append(prot)
    return T

nuc_idx = dict((c,i) for i,c in enumerate(dna_alphabet))
aa_idx = dict((c,i) for i,c in enumerate(aa_alphabet))

def to_idx(seq, idx = aa_idx): # used later to one-hot encode
    return np.array(list(idx[c] for c in seq))

def to_aa_seq(profile, aa_alphabet = aa_alphabet):
    assert len(profile.shape) == 2
    assert profile.shape[1] == 21
    aaseq = ""
    for i in range(profile.shape[0]):        
        aa = aa_alphabet[ np.argmax(profile[i,:])+1 ]
        aaseq += aa
        
    return aaseq

def makeDFs(P):
    (k, s, u) = P.shape
    dfs = []
    for j in range(u):
        profile_matrix = P[:,:,j]
        df = pd.DataFrame(profile_matrix)
        df.columns = aa_alphabet[1:]
        df = df.drop(['*'], axis=1)
        dfs.append(df)
    return dfs

