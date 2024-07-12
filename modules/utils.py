# general useful stuff

import numpy as np

def full_stack():
    """ Call this in an exception clause to output the full stack trace of an exception. """
    import traceback, sys
    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]       # remove call of full_stack, the printed exception
                            # will contain the caught exception caller instead
    trc = 'Traceback (most recent call last):\n'
    stackstr = trc + ''.join(traceback.format_list(stack))
    if exc is not None:
        stackstr += '  ' + traceback.format_exc().lstrip(trc)
    return stackstr


# === DNA translation stuff ============================================================================================

_dna_alphabet = "ACGTacgtNWSMKRYBDHVNZ " # for real sequences, need to know about softmask and ambiguous bases and gaps
_complements =  "TGCAtgcaNNNNNNNNNNNNN " # also map softmask, map ambiguous to N, keep padding (gaps)
_rctbl = str.maketrans(_dna_alphabet, _complements)

_codon_len = 3
_genetic_code = { # translation table 1 of NCBI
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


def six_frame_translation(S: str):
    """ return all 6 conceptually translated protein sequences """
    T = []
    for seq in (S, S[::-1].translate(_rctbl)): # forward, reverse-complement sequence
        for f in range(3): # frame
            prot = ""
            for i in range(f, len(S) - _codon_len + 1, _codon_len):
                codon = seq[i:i+_codon_len]
                if codon not in _genetic_code: # real sequences may contain N's or softmasking or ambiguous bases
                    prot += ' '                    # use null aa in that case
                else:
                    prot += _genetic_code[codon]
                
            T.append(prot)
    return T


def sequence_translation(S: str, rc = False):
    """ Translate single DNA sequence to AA sequence. Set `rc` to `True` to translate the reverse complement of `S` """
    if rc:
        S = S[::-1].translate(_rctbl)
        
    prot = ""
    for i in range(0, len(S)-3+1, 3):
        codon = S[i:i+3]
        if codon not in _genetic_code: # real sequences may contain N's or softmasking or ambiguous bases
            prot += ' '                # use null aa in that case
        else:
            prot += _genetic_code[codon]
            
    return prot


# more stuff

def oneHot(seq, alphabet): # one hot encoding of a sequence
    oh = np.zeros((len(seq), len(alphabet)))
    for i, c in enumerate(seq):
        if c in alphabet:
            oh[i,alphabet.index(c)] = 1.0
    return oh