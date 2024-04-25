# Utility functions for DNA and amino acid sequences
# Mario Stanke, August 2021
import numpy as np
import pandas as pd

dna_alphabet = "ACGTacgtNWSMKRYBDHVNZ " # for real sequences, need to know about softmask and ambiguous bases and gaps (from padding)
complements =  "TGCAtgcaNNNNNNNNNNNNN " # also map softmask, map ambiguous to N, keep padding
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

def sequence_translation(S, rc = False):
    """ Translate single DNA sequence to AA sequence. Set `rc` to `True` to translate the reverse complement of `S` """
    if rc:
        S = S[::-1].translate(rctbl)
        
    prot = ""
    for i in range(0, len(S)-3+1, 3):
        codon = S[i:i+3]
        if codon not in genetic_code: # real sequences may contain N's or softmasking or ambiguous bases
            prot += ' '               # use null aa in that case
        else:
            prot += genetic_code[codon]
            
    return prot
    
    
    
def three_frame_translation(S, rc = False, offsets=range(3)):
    """ Translate single DNA sequence to three AA sequences, one per reading frame. Reading frames can be customized by
          setting `offsets` to a list of offsets (default: [0, 1, 2]).
        Set `rc` to `True` to translate the reverse complement of `S` """
    T = []
    if rc:
        S = S[::-1].translate(rctbl)

    for f in offsets: # frame
        prot = ""
        for i in range(f, len(S)-3+1, 3):
            codon = S[i:i+3]
            if codon not in genetic_code: # real sequences may contain N's or softmasking or ambiguous bases
                prot += ' '               # use null aa in that case
            else:
                prot += genetic_code[codon]
        T.append(prot)

    return T

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
                
            T.append(prot)
    return T

def convert_six_frame_position(pos: int, frame_idx: int, dna_seqlen: int, dna_to_aa: bool=True):
    """ Convert a sequence position according to six_frame_translation(),
          i.e. either from the nucleotide sequence to a positon in the aa sequence according to frame_idx
          or the other way around (dna_to_aa False). In the latter case, the returned position always refers to the
          _first codon base w.r.t. the forward strand_.
          Example: g = 'AAACCCTTTGGG' translates to a0 = 'KPFG', a1 = 'NPL', a2 = 'TLW', a3 = 'PKGF', a4 = 'PKG' and
           a5 =  'QRV' (a0-a2: forward, a3-a5: reverse complement).
           a0[0] (K) then maps to g[0] ([A]AA...), a1[0] (N) -> g[1] (A[A]AC..), a2[0] (T) -> g[2] (AA[A]CC...),
           and a3[0] (P) -> g[9] (...TT[G]GG), a4[0] -> g[8] (...TT[T]GGG), a5[0] (Q) -> g[7] (...CT[T]TGGG), etc.
          
        Parameters:
            pos (int): position to be converted
            frame_idx (int): frame index as returned from six_frame_translation, must be in [0,1,2,3,4,5]
            dna_seqlen (int): length of the dna sequence
            dna_to_aa (bool): direction of conversion
    """
    assert frame_idx in range(0,6), "[ERROR] >>> Invalid frame_idx "+str(frame_idx)+", must be a value from 0 to 5"
    assert pos >= 0, "[ERROR] >>> Invalid position "+str(pos)+", must be positive"
    
    if dna_to_aa:
        if frame_idx >= 3:
            pos = dna_seqlen - 1 - pos
            frame_idx -= 3
            
        if pos < frame_idx:
            return -1 # before start of sequence translation
        else:
            return (pos-frame_idx) // 3
    
    else:
        convPos = (pos*3)+frame_idx
        if frame_idx >= 3:
            convPos -= 3 # correct frame shift
            convPos = dna_seqlen-1-convPos # count from back
            convPos -= 2 # first codon base w.r.t. forward strand
            
        return convPos
            
            

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

def makeDFs(P, alphabet:list[str] = aa_alphabet[1:], drop:list[str] = ['*']):
    """ Create a list of pandas DataFrames from a profile matrix P.
        Each DataFrame corresponds to a profile in the third dimension of P.
        
        Parameters:
            P (np.ndarray): profile matrix of shape (k, alphabet_size, u (number of profiles))
            alphabet (list[str]): list of characters for the columns of the DataFrames. Default: aa_alphabet[1:]
            drop (list[str]): list of characters to drop from the columns of the DataFrames. Default: ['*']
        
        Returns:
            list[pd.DataFrame]: list of DataFrames
    """
    (k, s, u) = P.shape
    dfs = []
    for j in range(u):
        profile_matrix = P[:,:,j]
        df = pd.DataFrame(profile_matrix)
        df.columns = alphabet
        for dc in drop:
            if dc in df.columns:
                df = df.drop([dc], axis=1)
        
        dfs.append(df)
    return dfs
