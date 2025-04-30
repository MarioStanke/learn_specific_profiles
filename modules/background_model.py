""" Module for implementations of higher order background models. """

from Bio import SeqIO
from pathlib import Path
import numpy as np

# human intergenic regions dinucleotide frequencies from Augustus
HG_INTERGENIC_DINT_FREQ = {
    'AA': 0.0916,
    'AC': 0.0506,
    'AG': 0.0717,
    'AT': 0.0693,
    'CA': 0.0728,
    'CC': 0.0586,
    'CG': 0.0137,
    'CT': 0.0717,
    'GA': 0.0596,
    'GC': 0.048,
    'GG': 0.0586,
    'GT': 0.0506,
    'TA': 0.0592,
    'TC': 0.0596,
    'TG': 0.0728,
    'TT': 0.0916
}


def _get_augustus_model():
    """
    Get the Augustus model for human intergenic regions. Returns a 4x4 matrix
    representing the model, a dict with dinucleotide frequencies and the alphabet used (ACGT).
    The matrix is constructed such that each row sums to 1, and the values
    are non-negative. The model is based on the dinucleotide frequencies
    from Augustus for human intergenic regions.
    """
    assert len(HG_INTERGENIC_DINT_FREQ) == 16, "Expected 16 dinucleotide frequencies"
    assert sum(HG_INTERGENIC_DINT_FREQ.values()) == 1.0, "Expected sum of dinucleotide frequencies to be 1.0"
    model = np.zeros((4, 4))
    alphabet = "ACGT"
    for i, a in enumerate(alphabet):
        s = sum(HG_INTERGENIC_DINT_FREQ[a + b] for b in alphabet)
        for j, b in enumerate(alphabet):
            model[i, j] = HG_INTERGENIC_DINT_FREQ[a + b] / s

        assert abs(sum(model[i, :]) - 1.0) < 1e-6, f"Expected sum of row {i} to be 1.0, got {sum(model[i, :])}"
        assert all(model[i, :] >= 0), f"Expected all values in row {i} to be non-negative, got {model[i, :]}"

    return model, HG_INTERGENIC_DINT_FREQ.copy(), alphabet


def _get_uniform_model(order: int):
    """
    Get a uniform background model for a given order. The model is a {4}^(order+1) matrix
    representing the uniform distribution of nucleotides. The alphabet used is ACGT.
    Returns a tuple of the model (np.ndarray) and the frequencies of the nucleotides (dict) 
    and the alphabet used (str 'ACGT').
    """
    assert order >= 0, "Order must be greater than or equal to 0"
    alphabet = "ACGT"
    size = len(alphabet)
    model = np.full((size,)*(1+order), 1.0 / size)
    if order == 0:
        freqs = {c: 1.0 / size for c in alphabet}
    else:
        freqs = {}
        for i in range(len(alphabet)**(order + 1)):
            kmer = "".join([alphabet[i // (len(alphabet)**j) % len(alphabet)] for j in range((order + 1))])
            freqs[kmer] = 1.0 / (len(alphabet)**(order + 1))
    return model, freqs, alphabet


def get_background_model(order: int, model_type: str = "uniform", src: Path = None):
    """
    Get a background model for a given order and model type. The model can be either
    'uniform', 'data' or 'augustus'. The uniform model is a uniform distribution of nucleotides,
    while the data model is based on the dinucleotide frequencies from a given file.
    The Augustus model is based on the dinucleotide frequencies from Augustus for
    human intergenic regions and is restricted to order 1.
    The function returns a tuple of the model (np.ndarray) and the frequencies of the nucleotides (dict)
    and the alphabet used (str 'ACGT').
    """
    assert order >= 0, "Order must be greater than or equal to 0"
    assert model_type in ["uniform", "data", "augustus"], "Model type must be either 'uniform', 'data' or 'augustus'"

    if model_type == "uniform":
        if src is not None:
            print(f"[Warning] >>> Source file {src} is ignored for uniform model")
        return _get_uniform_model(order)
    elif model_type == "augustus":
        assert order == 1, "Order must be 1 for Augustus model"
        if src is not None:
            print(f"[Warning] >>> Source file {src} is ignored for uniform model")
        return _get_augustus_model()
    elif model_type == "data":
        alphabet = "ACGT"
        assert src is not None, "Source file must be provided for data model"
        assert src.exists(), f"Source file {src} does not exist"
        records = [r for r in SeqIO.parse(src, "fasta")]
        assert len(records) > 0, f"Source file {src} is empty or contains no records"
        assert all(len(r.seq) > 0 for r in records), f"Source file {src} contains empty sequences"
        freqs = {}
        k = order + 1
        # it's possible that not all k-mers are present in the sequences, so we need to
        # initialize the frequencies to 0
        if order == 0:
            freqs = {c: 0 for c in alphabet}
        else:
            for i in range(len(alphabet)**k):
                kmer = "".join([alphabet[i // (len(alphabet)**j) % len(alphabet)] for j in range(k)])
                freqs[kmer] = 0
        # count the frequencies of k-mers in the sequences
        for record in records:
            seq = str(record.seq)
            for i in range(len(seq) - k):
                kmer = seq[i:i + k]
                # ignore ambiguous or softmasked nucleotides
                if all(c in alphabet for c in kmer):    
                    assert kmer in freqs, f"Expected k-mer {kmer} to be in frequencies dictionary"
                    freqs[kmer] += 1
        # normalize frequencies
        total = sum(freqs.values())
        if total == 0:
            print(f"[Warning] >>> No {k=}-mers found in sequences, using uniform model")
            return _get_uniform_model(order)
        assert total > 0, f"Expected total frequency to be greater than 0, got {total}"
        # normalize frequencies to sum to 1
        for kmer in freqs:
            freqs[kmer] /= total
        # create a matrix of shape {4}^(order+1)
        size = len(alphabet)
        model = np.zeros((size,)*k)
        # print(f"[DEBUG] >>> {order=}, {k=}, {size=}, {model.shape=}")
        if order == 0:
            # 0-mers, i.e. the frequencies of the nucleotides
            for i in range(size):
                model[i] = freqs[alphabet[i]]
        else:
            for i in range(len(alphabet)**order):
                # k-1-mers, i.e. the first #order nucleotides -> add the last nucleotide later to handle missing k-mers
                o_nt = "".join([alphabet[i // (len(alphabet)**j) % len(alphabet)] for j in range(order)])
                o_idcs = tuple([alphabet.index(c) for c in o_nt])
                s = sum(freqs[o_nt + alphabet[i]] for i in range(size))
                # print(f"[DEBUG] >>> {o_nt=}, {o_idcs=}, {s=}")
                for i in range(size):
                    kmer = o_nt + alphabet[i]
                    idcs = o_idcs + (i,)
                    if s == 0:
                        model[idcs] = 1/size # k-mer was not found in the sequences, use uniform distribution
                    else:
                        model[idcs] = freqs[kmer] / s
                # print(f"[DEBUG] >>> {o_idcs=}, {model=}, {model[o_idcs][:]=}, {model[0, :]=}, {sum(model[o_idcs, :])=}")
                assert abs(sum(model[o_idcs][:]) - 1.0) < 1e-6, \
                    f"Expected sum of row {o_idcs} to be 1.0, got {sum(model[o_idcs][:])}"
                assert all(model[o_idcs][:] >= 0), \
                    f"Expected all values in row {o_idcs} to be non-negative, got {model[o_idcs][:]}"
        
        return model, freqs, alphabet