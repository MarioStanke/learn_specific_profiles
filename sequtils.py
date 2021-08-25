# Utility functions for DNA and amino acid sequences
# Mario Stanke, August 2021
import numpy as np

dna_alphabet = "ACGT"
complements = "TGCA"
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
aa_alphabet_size = len(aa_alphabet)

def six_frame_translation(S):
    """ return all 6 conceptually translated protein sequences """
    T = []
    for seq in (S, S[::-1].translate(rctbl)): # forward, reverse-complement sequence
        for f in range(3): # frame
            prot = ""
            for i in range(f, len(S) - codon_len + 1, codon_len):
                prot += genetic_code[seq[i:i+codon_len]]
            T.append(prot)
    return T

nuc_idx = dict((c,i) for i,c in enumerate(dna_alphabet))
aa_idx = dict((c,i) for i,c in enumerate(aa_alphabet))

def to_idx(seq, idx = aa_idx): # used later to one-hot encode
    return np.array(list(idx[c] for c in seq))
