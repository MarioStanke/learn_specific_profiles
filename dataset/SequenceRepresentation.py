# ME 20230515

"""
This module contains classes for representing sequences and genomic elements.
"""

import json
import os

class Sequence:
    """ Basic sequence class. Sequence is viewed as a subsequence of a larger sequence, i.e. a chromosome. 
        It can also have a homology relation with another sequence. It can contain numerous genomic elements, that again
        can have homology relations with other sequences. Genomic elements and homologies are themselves Sequence 
        objects. 
        
        Attributes:
            id (str): Unique identifier of the sequence, resembles genome browser range 
                        (e.g. Homo_sapiens:chr1:1,000-1,100)
            species (str): Species name.
            chromosome (str): Chromosome (or scaffold, ...) name.
            strand (str): Strand of the sequence. Can be either '+' or '-'.
            genome_start (int): Start position of the sequence on the chromosome.
            genome_end (int): End position of the sequence on the chromosome.
            lenght (int): Lenght of the sequence.
            sequence (str): The sequence itself, might be None.
            type (str): Description of sequence type, e.g. `sequence` or `exon`
            homology (list): List of homology relations to other sequences, might be None.
            genomic_elements (list): List of genomic elements, might be None.
        """
    
    def __init__(self, species, chromosome, strand, genome_start, genome_end = None, lenght = None, sequence = None,
                 seqtype = 'sequence', no_homology = False, no_elements = False) -> None:
        """ Initialize a sequence object. 
            Positions are 0-based, i.e. the first position is 0. The end position is not included in the sequence.
            Example: We have a "chromosome" `AAATTTAAA` and want to store the sequence `TTT`. Then we have to set
            `genome_start = 3` and `genome_end = 6`.

            Args:
                species (str): Species name.
                chromosome (str): Chromosome (or scaffold, ...) name.
                strand (str): Strand of the sequence. Can be either '+' or '-'.
                genome_start (int): Start position of the sequence on the chromosome.

                At least one of the following optional arguments must be provided:
                genome_end (int): End position of the sequence on the chromosome.
                lenght (int): Lenght of the sequence.
                sequence (str): The sequence itself.

                seqtype (str): Description of sequence type, e.g. `sequence` (default) or `exon`
                no_homology (bool): If True, no homology relation is stored. Use this for genomic elements of the 
                                      reference genome.
                no_elements (bool): If True, no genomic elements are stored. Use this for genomic elements.
        """

        assert strand in ['+', '-'], "[ERROR] >>> Strand must be either '+' or '-'."
        assert genome_start >= 0, "[ERROR] >>> Start position must be a positive integer."
        self.species = species
        self.chromosome = chromosome
        self.strand = strand
        self.genome_start = genome_start
        self.type = seqtype

        assert not (genome_end is None and lenght is None and sequence is None), \
            "[ERROR] >>> At least one of the following arguments must be provided: genome_end, lenght, sequence."
        if genome_end is not None:
            assert genome_end >= genome_start, "[ERROR] >>> End position must be greater than start position."
            self.genome_end = genome_end
            self.lenght = genome_end - genome_start
            if lenght is not None:
                assert lenght == genome_end - genome_start, \
                    "[ERROR] >>> Lenght must be equal to end position minus start position."
            if sequence is not None:
                self.sequence = sequence
                assert self.lenght == len(sequence), \
                    "[ERROR] >>> Lenght must be equal to length of the sequence."
        else:
            if lenght is not None:
                assert lenght > 0, "[ERROR] >>> Lenght must be a positive integer."
                self.lenght = lenght
                self.genome_end = genome_start + lenght
                if sequence is not None:
                    self.sequence = sequence
                    assert self.lenght == len(sequence), \
                        "[ERROR] >>> Lenght must be equal to length of the sequence."
            else:
                assert sequence is not None, "[ERROR] >>> Sequence must be a string." # should always be true
                self.lenght = len(sequence)
                self.genome_end = genome_start + len(sequence)
                self.sequence = sequence

        assert hasattr(self, 'genome_end'), "[ERROR] >>> `genome_end` not set."
        assert hasattr(self, 'lenght'), "[ERROR] >>> `lenght` not set."

        self.id = f"{self.species}:{self.chromosome}:{self.genome_start:,}-{self.genome_end:,}"

        if not hasattr(self, 'sequence'):
            self.sequence = None

        if not no_homology:
            self.homology = []
        if not no_elements:
            self.genomic_elements = [] # store exons etc. later

    def __str__(self) -> str:
        """ String representation of the sequence. """
        rep = f"{self.id} ({self.strand}, length = {self.lenght:,}, type = {self.type})"
        if self.sequence is not None:
            if self.lenght > 23:
                rep += f"\n{self.sequence[:10]}...{self.sequence[-10:]}"
            else:
                rep += f"\n{self.sequence}"
        if hasattr(self, 'genomic_elements'):
            rep += f"\n{len(self.genomic_elements)} genomic elements"
        if hasattr(self, 'homology'):
            rep += f"\n{len(self.homology)} homologies"

        return rep
    
    def __repr__(self) -> str:
        """ Representation of the sequence. This is a JSON string. """
        return json.dumps(self.toDict(), indent = 4)
    
    def __len__(self) -> int:
        """ Length of the sequence. """
        return self.lenght
    
    def addElement(self, element):
        """ Add a genomic element to the sequence. """
        _addElementToSequence(element, self)

    def addSubsequenceAsElement(self, start: int, end: int, seqtype: str, strand: str = None, 
                                genomic_positions: bool = False, **kwargs):
        """ Define a subsequence of the sequence and add it as a genomic element. 
            Args:
                start (int): Start position of the subsequence.
                end (int): End position of the subsequence.
                seqtype (str): Description of element type, e.g. `exon`
                strand (str): Strand of the subsequence. Can be either '+' or '-'. If None, the strand of the sequence
                                is used.
                genomic_positions (bool): If True, the positions are interpreted as genomic positions, otherwise as
                                            positions within the sequence.
                **kwargs: Additional arguments for the element, i.e. `no_homology` and `no_elements`
        """
        if genomic_positions:
            assert start >= self.genome_start, "[ERROR] >>> `start` must be greater than or equal to `genome_start`."
            assert end <= self.genome_end, "[ERROR] >>> `end` must be less than or equal to `genome_end`."
            assert start <= end, "[ERROR] >>> `start` must be less than or equal to `end`."
            if strand is None:
                strand = self.strand

            if self.sequence is not None:
                sequence = self.sequence[start - self.genome_start:end - self.genome_start]
            else:
                sequence = None

            element = Sequence(self.species, self.chromosome, strand, start, end, sequence=sequence, seqtype=seqtype, 
                               **kwargs)
            self.addElement(element)

        else:
            assert start >= 0, "[ERROR] >>> `start` must be greater than or equal to 0."
            assert end <= self.lenght, "[ERROR] >>> `end` must be less than or equal to `lenght`."
            assert start <= end, "[ERROR] >>> `start` must be less than or equal to `end`."
            if strand is None:
                strand = self.strand

            if self.sequence is not None:
                sequence = self.sequence[start:end]
            else:
                sequence = None

            element = Sequence(self.species, self.chromosome, strand, self.genome_start+start, self.genome_end+end, 
                               sequence=sequence, seqtype=seqtype, **kwargs)
            self.addElement(element)

    def addHomology(self, homology):
        """ Add a homology to the sequence. """
        _addHomologyToSequence(homology, self)

    def toDict(self) -> dict:
        """ Return a dictionary representation of the sequence. """
        objdict = {
            'id': self.id,
            'species': self.species,
            'chromosome': self.chromosome,
            'strand': self.strand,
            'genome_start': self.genome_start,
            'genome_end': self.genome_end,
            'lenght': self.lenght
        }
        if self.sequence is not None:
            objdict['sequence'] = self.sequence

        objdict['type'] = self.type

        if hasattr(self, 'genomic_elements'):
            objdict['genomic_elements'] = [element.toDict() for element in self.genomic_elements]
        if hasattr(self, 'homology'):
            objdict['homology'] = [homology.toDict() for homology in self.homology]

        return objdict



# helper functions for adding genomic elements and homology to a sequence
def _addElementToSequence(element: Sequence, sequence: Sequence):
    """ Helper function for adding genomic elements to a sequence, do not use directly """
    assert type(element) == Sequence, "[ERROR] >>> `element` must be of type Sequence."
    assert type(sequence) == Sequence, "[ERROR] >>> `sequence` must be of type Sequence."
    assert hasattr(sequence, 'genomic_elements'), "[ERROR] >>> `sequence` must have attribute `genomic_elements`."
    assert element.genome_start >= sequence.genome_start, "[ERROR] >>> `element` must start after `sequence`."
    assert element.genome_end <= sequence.genome_end, "[ERROR] >>> `element` must end before `sequence`."
    sequence.genomic_elements.append(element)

def _addHomologyToSequence(homology: Sequence, sequence: Sequence):
    """ Helper function for adding a homology to a sequence, do not use directly """
    assert type(homology) == Sequence, "[ERROR] >>> `homology` must be of type Sequence."
    assert type(sequence) == Sequence, "[ERROR] >>> `sequence` must be of type Sequence."
    assert hasattr(sequence, 'homology'), "[ERROR] >>> `sequence` must have attribute `homology`."
    sequence.homology.append(homology)

# create a Sequence object from a JSON file or string
def fromJSON(jsonfile: str = None, jsonstring: str = None) -> Sequence:
    """ Create a Sequence object from a JSON file or a JSON string. 
        If both arguments are provided, the file will be used. """
    assert not (jsonfile is None and jsonstring is None), \
        "[ERROR] >>> Either `jsonfile` or `jsonstring` must be provided."
    if jsonfile is not None:
        assert os.path.isfile(jsonfile), "[ERROR] >>> `jsonfile` must be a valid file."
        with open(jsonfile, 'rt') as f:
            objdict = json.load(f)
    else:
        objdict = json.loads(jsonstring)

    sequence = Sequence(
        species = objdict['species'],
        chromosome = objdict['chromosome'],
        strand = objdict['strand'],
        genome_start = objdict['genome_start'],
        genome_end = objdict['genome_end'],
        lenght = objdict['lenght'],
        seqtype = objdict['type']
    )
    if 'sequence' in objdict:
        sequence.sequence = objdict['sequence']
    if 'genomic_elements' in objdict:
        for element in objdict['genomic_elements']:
            sequence.addElement(fromJSON(jsonstring = json.dumps(element)))
    if 'homology' in objdict:
        for homology in objdict['homology']:
            sequence.addHomology(fromJSON(jsonstring = json.dumps(homology)))

    return sequence