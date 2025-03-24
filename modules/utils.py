# general useful stuff

import gzip
import numpy as np
import pandas as pd
from pathlib import Path
import re

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


# PAM matrix

def read_pam(aa_order: str, path = None) -> tuple[np.ndarray, np.ndarray]:
    """ Read a PAM matrix from a file. If path is None (default), return the default rate matrix. The row order is 
     determined by `aa_order`, which must be an iterable containing exactly all 20 amino acids. 

     Returns:
            A symmetric exchangeability matrix with zero diagonal and a frequency vector. """
    if path is not None:
        raise NotImplementedError("Reading PAM matrices from files is not implemented yet.")
    
    def parse_paml(lines: list[str]):
        """ Parses the content of a paml file. """
        # the first 19 lines of LG_paml contain the exchangeability matrix, the last line the amino acid frequencies
        paml_alphabet = "A R N D C Q E G H I L K M F P S T W Y V".split(" ")
        assert all([aa in paml_alphabet for aa in aa_order]), \
            f"[ERROR] >>> Invalid amino acid order: {aa_order} vs. {paml_alphabet}"
        assert all([aa in aa_order for aa in paml_alphabet]), \
            f"[ERROR] >>> Invalid amino acid order: {aa_order} vs. {paml_alphabet}"
        s = len(paml_alphabet)
        R = np.zeros((s, s), dtype=np.float32)
        for i in range(1,s):
            R[i,:i] = R[:i,i] = np.fromstring(lines[i-1], sep=" ") 
        p = np.fromstring(lines[s-1], sep=" ", dtype=np.float32)
        #reorganize to match the amino acid order in desired_alphabet
        perm = [paml_alphabet.index(aa) for aa in aa_order]
        p = p[perm]
        R = R[perm, :]
        R = R[:, perm]
        return R, p

    # the default rate matrix ("LG")
    LG_paml = ['0.425093 \n', 
               '0.276818 0.751878 \n', 
               '0.395144 0.123954 5.076149 \n', 
               '2.489084 0.534551 0.528768 0.062556 \n', 
               '0.969894 2.807908 1.695752 0.523386 0.084808 \n', 
               '1.038545 0.363970 0.541712 5.243870 0.003499 4.128591 \n', 
               '2.066040 0.390192 1.437645 0.844926 0.569265 0.267959 0.348847 \n', 
               '0.358858 2.426601 4.509238 0.927114 0.640543 4.813505 0.423881 0.311484 \n', 
               '0.149830 0.126991 0.191503 0.010690 0.320627 0.072854 0.044265 0.008705 0.108882 \n', 
               '0.395337 0.301848 0.068427 0.015076 0.594007 0.582457 0.069673 0.044261 0.366317 4.145067 \n', 
               '0.536518 6.326067 2.145078 0.282959 0.013266 3.234294 1.807177 0.296636 0.697264 0.159069 0.137500 \n',
               '1.124035 0.484133 0.371004 0.025548 0.893680 1.672569 0.173735 0.139538 0.442472 4.273607 6.312358 0.656604 \n', 
               '0.253701 0.052722 0.089525 0.017416 1.105251 0.035855 0.018811 0.089586 0.682139 1.112727 2.592692 0.023918 1.798853 \n', 
               '1.177651 0.332533 0.161787 0.394456 0.075382 0.624294 0.419409 0.196961 0.508851 0.078281 0.249060 0.390322 0.099849 0.094464 \n', 
               '4.727182 0.858151 4.008358 1.240275 2.784478 1.223828 0.611973 1.739990 0.990012 0.064105 0.182287 0.748683 0.346960 0.361819 1.338132 \n', 
               '2.139501 0.578987 2.000679 0.425860 1.143480 1.080136 0.604545 0.129836 0.584262 1.033739 0.302936 1.136863 2.020366 0.165001 0.571468 6.472279 \n', 
               '0.180717 0.593607 0.045376 0.029890 0.670128 0.236199 0.077852 0.268491 0.597054 0.111660 0.619632 0.049906 0.696175 2.457121 0.095131 0.248862 0.140825 \n', 
               '0.218959 0.314440 0.612025 0.135107 1.165532 0.257336 0.120037 0.054679 5.306834 0.232523 0.299648 0.131932 0.481306 7.803902 0.089613 0.400547 0.245841 3.151815 \n', 
               '2.547870 0.170887 0.083688 0.037967 1.959291 0.210332 0.245034 0.076701 0.119013 10.649107 1.702745 0.185202 1.898718 0.654683 0.296501 0.098369 2.188158 0.189510 0.249313 \n', 
               '0.079066 0.055941 0.041977 0.053052 0.012937 0.040767 0.071586 0.057337 0.022355 0.062157 0.099081 0.064600 0.022951 0.042302 0.044040 0.061197 0.053287 0.012066 0.034155 0.069147 \n']
    
    return parse_paml(LG_paml)


# more stuff

def oneHot(seq, alphabet): # one hot encoding of a sequence
    oh = np.zeros((len(seq), len(alphabet)))
    for i, c in enumerate(seq):
        if c in alphabet:
            oh[i,alphabet.index(c)] = 1.0
    return oh


# read a *.meme file and extract the PWM in there
def readMemeFile(file: Path, motif_filter: list[str] = None, alph: str = 'ACGT') -> dict[str, np.ndarray]:
    """ Read a file in meme format and extract the PWM from it as a list of numpy arrays of shape (w, a) where `w` is 
      the width of the motiv and `a` is the alphabet size (4 by default). All motifs in the file must have the same
      alphabet, and the alphabet must be given as second argument as a single string (default: 'ACGT').

      If there are multiple motifs of the same name in the file, only the first occurrence will be parsed

      You can also provide a list of motif names to filter for, in which case only the motifs with these names will be
        returned.

    Returns:
        dict[str, np.ndarray] (map from motif name to PWM)
    """
    # parse reference motif PWMs
    assert file.exists(), f"[utils.readMemeFile] Error: no file `{file}` found."
    motifs = {}
    with open(file) as f:
        alphabet_seen = False
        motif_name_seen = False
        motif_started = False
        motif_finished = True
        motif_site = None
        for line in f:
            line = line.strip()
            if line.startswith('ALPHABET'):
                m = re.match(r'ALPHABET= ([A-Z]+)', line)
                assert m, f"[utils.readMemeFile] Error: no valid alphabet in line '{line}'"
                alphabet = m.group(1)
                assert alphabet == alph, f"[utils.readMemeFile] Error: require alphabet '{alph}', but seen '{alphabet}'"
                alphabet_seen = True

            if line.startswith('MOTIF'):
                assert alphabet_seen, f"[utils.readMemeFile] Error: encountered motif before alphabet in '{line}'"
                assert motif_finished, \
                    f"[utils.readMemeFile] Error: encountered new motif before the last was finished in '{line}'"
                assert motif_site is None, \
                    f"[utils.readMemeFile] Error: encountered new motif before the last was finished in '{line}'"
                motif_id = line.split()[1]
                motif_name_seen = True
                motif_started = False
                motif_finished = False

                if (motif_id in motifs) or (motif_filter is not None and motif_id not in motif_filter):
                    # reset flags to skip to next motif if this motif name was already encountered or not in the filter
                    motif_name_seen = False
                    motif_started = False
                    motif_finished = True

            if motif_name_seen:
                if line.startswith('letter-probability matrix'):
                    m = re.match(r'letter-probability matrix: alength= 4 w= (\d+).+', line)
                    assert m, f"[utils.readMemeFile] Error: expected motif details, got '{line}'"
                    motif_length = int(m.group(1))
                    motif = np.zeros((motif_length, len(alph)))
                    motif_started = True
                    motif_site = 0
                else:
                    if not motif_started:
                        continue
                    elif motif_started and re.match(r'^\s*$', line):
                        # done parsing motif, reset flags
                        motif_name_seen = False
                        motif_started = False
                        motif_finished = True
                        motif_site = None
                        motifs[motif_id] = motif
                    else:
                        assert motif_site is not None, \
                            f"[utils.readMemeFile] Error: flag motif_site not set when encountering line '{line}'"
                        motif[motif_site] = list(map(float, line.split()))
                        motif_site += 1

    return motifs


def getMotifRC(pwm: np.ndarray, alph: str = 'ACGT', alph_comp: str = "TGCA") -> np.ndarray:
    """ Compute the reverse complement of a single motif PWM as returned by readMemeFile(). Needs the alphabet of the
    input pwm as well as the corresponding complement alphabet, 
    i.e. alph_comp[i] must be the complement character of alph[i]. """
    assert len(alph) == len(alph_comp), \
        f"[utils.readMemeFile] Error: Alphabet sizes differ ({len(alph)} != {len(alph_comp)})"
    assert sorted(alph) == sorted(alph_comp), \
        f"[utils.readMemeFile] Error: Alphabet contents differ ({alph} != {alph_comp})"
    assert pwm.shape[1] == len(alph), \
        f"[utils.readMemeFile] Error: Alphabet size {len(alph)} does not match PWM shape {pwm.shape}"

    rc = np.zeros(pwm.shape)
    motif_len = pwm.shape[0]
    cmap = {i: alph.find(alph_comp[i]) for i in range(len(alph))}
    for i in range(motif_len):
        j = motif_len-1 - i # fwd position
        for k in range(len(alph)):
            rc[j,cmap[k]] = pwm[i,k]

    return rc


# load a BED file and return it as a Pandas DataFrame
def readBEDFile(filepath: Path) -> pd.DataFrame:
    """ Load a BED file and return its content as a pd.DataFrame

    Args:
        filepath: Path - pathlib.Path object pointing to the file. If the filepath ends in `.gz`, a gzipped file is
                         assumed and unzipped on loading

    Returns:
        pd.DataFrame with columns ['chrom', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 
        'qValue', 'peak']
    """

    # file format:
    # https://genome.ucsc.edu/FAQ/FAQformat.html#format12
    # chrom - Name of the chromosome (or contig, scaffold, etc.).
    # chromStart - The starting position of the feature in the chromosome or scaffold. The first base in a chromosome is 
    #              numbered 0.
    # chromEnd - The ending position of the feature in the chromosome or scaffold. The chromEnd base is not included in 
    #            the display of the feature.
    # name - Name given to a region (preferably unique). Use "." if no name is assigned.
    # score - Indicates how dark the peak will be displayed in the browser (0-1000). If all scores were '0' when the 
    #         data was submitted to the DCC, the DCC assigned scores 1-1000 based on signal value. Ideally the average 
    #         signalValue per base spread is between 100-1000.
    # strand - +/- to denote strand or orientation (whenever applicable). Use "." if no orientation is assigned.
    # signalValue - Measurement of overall (usually, average) enrichment for the region.
    # pValue - Measurement of statistical significance (-log10). Use -1 if no pValue is assigned.
    # qValue - Measurement of statistical significance using false discovery rate (-log10). Use -1 if no qValue is 
    #          assigned.
    # peak - Point-source called for this peak; 0-based offset from chromStart. Use -1 if no point-source called.

    assert filepath.exists(), f"[utils.readBEDFile] Error: no file `{filepath}` found."
    if filepath.suffix == '.gz':
        openf = gzip.open
    else:
        openf = open

    with openf(filepath, 'r') as f:
        df = pd.read_csv(f, sep='\t', header=None)
        df.columns = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 
                      'peak']
        # require peak to be called
        assert not any(df['peak'] < 0), \
            f"[utils.readBEDFile] Error: {filepath.name}: {[str(t) for t in df.itertuples() if t.peak < 0][0]}" 

        return df