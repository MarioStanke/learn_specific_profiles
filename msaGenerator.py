import math
import os
import random
from turtle import right
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pyvolve



# ---------------
# tree generators
# ---------------

def star_tree(num_leaves:int, height:float):
    '''
    Generate a newick string for a star-like tree.
        Parameters:
            num_leaves (int): number of leaves
            height (float): time from ancestor to each of the leaves
        Returns:
            newick (str): Newick-formatted string, e.g. (seq1:0.1,seq2:0.1,seq3:0.1,seq4:0.1);
    '''
    newick = "(" + ",".join([f"seq{i}:{height}" for i in range(1, num_leaves+1)]) + ");"
    return newick



def caterpillar_tree(num_leaves:int, height:float):
    '''
    Generate a newick string for a caterpillar tree (all internal nodes have a leaf as son)
        Parameters:
            num_leaves (int): number of leaves
            height (float): time from MRCA to each of the leaves
        Returns:
            newick (str): Newick-formatted string, e.g. (seq1:0.1,seq2:0.1,seq3:0.1,seq4:0.1);
    '''
    assert num_leaves > 1, "catterpillar_tree can only be constructed with at least 2 leaves"
    min_branch_len = height / (num_leaves-1)
    newick = "(" * (num_leaves-1) + \
        f"seq1:{min_branch_len:.3f}" + \
        "".join([f",seq{i}:{min_branch_len*(i-1):.3f}):{min_branch_len:.3f}" for i in range(2, num_leaves)]) + \
        f",seq{num_leaves}:{height:.3f});"
    return newick



# ---------------------------------------
# sequence generator and helper functions
# ---------------------------------------

def simulate_msa_impl(fa_fname, model, tree:str, sites, root_sequence):
    """ Sample a multiple sequence alignment given the tree and model """
    evtree = pyvolve.read_tree(tree = tree)
    if (root_sequence):
        partition = pyvolve.Partition(models = model, root_sequence = root_sequence)
    else:
        partition = pyvolve.Partition(models = model, size = sites)
    evolver = pyvolve.Evolver(partitions = partition, tree = evtree)
    evolver(seqfile = fa_fname, ratefile = None, infofile = None)



def simulate_msa(model, tree:str, sites=None, root_sequence=None):
    """ Wrapper that takes care of temporary fasta file handling """
    # get random temporary filename, cover the unlikely event that the file already exists
    while True:
        fname = ''.join(random.choices("1234567980ABCDEF", k=6)) + "_msaGenerator_tmp.fa"
        if not os.path.exists(fname):
            break

    # generate sequences
    simulate_msa_impl(fname, model, tree, sites, root_sequence)
    assert os.path.exists(fname) and os.path.isfile(fname), "Something went wrong generating the MSA"

    # load sequences and delete tmp file
    sequences = []
    for seq in SeqIO.parse(fname, 'fasta'):
        sequences.append(seq)

    os.remove(fname) # cleanup
    return sequences



def concatenate_sequences(sequences, new_parts):
    """ Concatenates the sequences in two sequence lists from simulate_msa, appending `new_parts` to `sequences` """
    assert len(sequences) == len(new_parts), "sequences and new_parts must have same number of elements"
    for i in range(len(sequences)):
        seq = sequences[i]
        new = new_parts[i]
        assert seq.id == new.id, "Sequence order must be the same in both lists"
        assert seq.name == new.name, "Sequence order must be the same in both lists"
        assert seq.description == new.description, "Sequence order must be the same in both lists"
        concat = SeqRecord(
            Seq(str(seq.seq)+str(new.seq)),
            id=seq.id,
            name=seq.name,
            description=seq.description
        )
        sequences[i] = concat



# -------------
# main function
# -------------

def generate_sequences(num_sequences:int, seqlen:int, genelen: int,
                       coding_dist = 0.2, noncoding_dist = 0.4):
    '''
    Simulates evolution of `num_sequences` sequences that share a common ancestor
        Parameters:
            num_sequences (int): number of sequences to generate
            seqlen (int): total lenght of each generated sequence
            genelen (int): length of the simulated ortholog gene inside the sequences, rounded up to a multiple of 3
        Returns:
            sequences (list(Bio.SeqRecord.SeqRecord)): list of generated sequences as Biopython SeqRecords
            posDict (dict): dict containing positions of the sequence elements
    '''
    assert num_sequences > 0, "num_sequences must be greater than zero"
    assert genelen >= 0, "genelen must be positive"
    genelen = 3*math.ceil(genelen/3)
    assert seqlen >= genelen, "seqlen cannot be shorter than genelen (genelen was adjusted to "+str(genelen)+")"
    if genelen > 0:
        assert seqlen >= 6+genelen, "seqlen must be at least 6+genelen (genelen was adjusted to "+str(genelen)+")"

    # generate 3 copies of the same tree with different scales
    # This is  because pyvolve cannot use the mutation rate as part of the model.
    tree_noncoding = caterpillar_tree(num_sequences, noncoding_dist)
    tree_coding = caterpillar_tree(num_sequences, coding_dist)
    tree_signal = caterpillar_tree(num_sequences, 0) # no mutations of start and stop codon

    # stationary distribution of codons
    pi_codon = [1/61.] * 61
    # stationary distribution of nuleotides in noncoding region
    pi_nuc = [1/4.] * 4
    # transition transversion rate ratio for both coding and noncoding model
    kappa = 3

    # Define a simple nucleotide model
    parameters = {"state_freqs" : pi_nuc, # equilibrium distribution
                  "kappa" : kappa} # transition/transversion rate ratio
    nuc_model = pyvolve.Model("nucleotide", parameters)

    # Define a Goldman Yang codon model
    # Codon models require you to specify a second argument to pyvolve.Model, a dictionary of parameters. You must 
    #   specify dN/dS using either "omega" (for the full ratio), or "beta" for dN and "alpha" for dS, as shown below. 
    #   Either dictionary would be acceptable.
    parameters = {"omega": 0.4, # codon regions under negative selection
                  "state_freqs" : pi_codon, # equilibrium distribution
                  "kappa" : kappa} # transition/transversion rate ratio
    codon_model = pyvolve.Model("GY", parameters)

    # store information about element positions
    posDict = {
        '5flank_start': None,
        '5flank_len': None,
        'start_codon': None,
        'cds_start': None,
        'cds_len': genelen,
        'stop_codon': None,
        '3flank_start': None,
        '3flank_len': None
    }

    # only gene
    if seqlen == 6+genelen:
        sequences = simulate_msa(nuc_model, tree_signal, root_sequence = "ATG") # start codon
        concatenate_sequences(sequences,
                            simulate_msa(codon_model, tree_coding, genelen/3))  # coding sequence
        concatenate_sequences(sequences,                                        # stop codon
                            simulate_msa(nuc_model, tree_signal, root_sequence = random.choice(["TAA", "TAG", "TGA"])))
        posDict['start_codon'] = 0
        posDict['cds_start'] = 3
        posDict['stop_codon'] = genelen+3

    # no gene
    elif genelen == 0:
        sequences = simulate_msa(nuc_model, tree_noncoding, seqlen) # 5' flanking region
        posDict['5flank_start'] = 0
        posDict['5flank_len'] = seqlen

    # flanked gene
    else:
        leftFlankLen = (seqlen-genelen-6) // 2 # -6 for the start and stop codon
        rightFlankLen = seqlen - genelen - 6 - leftFlankLen

        # generate the 5 alignments
        sequences = simulate_msa(nuc_model, tree_noncoding, leftFlankLen) # 5' flanking region
        concatenate_sequences(sequences,
                            simulate_msa(nuc_model, tree_signal, root_sequence = "ATG")) # start codon
        concatenate_sequences(sequences,
                            simulate_msa(codon_model, tree_coding, genelen/3)) # coding sequence
        concatenate_sequences(sequences,                                         # stop codon
                            simulate_msa(nuc_model, tree_signal, root_sequence = random.choice(["TAA", "TAG", "TGA"])))
        concatenate_sequences(sequences,
                            simulate_msa(nuc_model, tree_noncoding, rightFlankLen)) # 3' flanking region
        posDict['5flank_start'] = 0
        posDict['5flank_len'] = leftFlankLen
        posDict['start_codon'] = leftFlankLen
        posDict['cds_start'] = leftFlankLen+3
        posDict['stop_codon'] = leftFlankLen+3+genelen
        posDict['3flank_start'] = leftFlankLen+3+genelen+3
        posDict['3flank_len'] = rightFlankLen

    
    return sequences, posDict
    