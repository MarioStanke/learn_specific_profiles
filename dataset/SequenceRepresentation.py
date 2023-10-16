# ME 20230515

"""
This module contains classes for representing sequences and genomic elements.
"""

from Bio.Seq import Seq
import json
import logging
import os
from typing import Union

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
            length (int): Length of the sequence.
            sequence (str): The top strand sequence itself, might be None.
            type (str): Description of sequence type, e.g. `sequence` or `exon`
            source (any): Optional reference to where the sequence comes from, useful for genomic elements
            homology (list): List of homology relations to other sequences, might be None.
            genomic_elements (list): List of genomic elements, might be None.
        """
    
    def __init__(self, species, chromosome, strand, genome_start, genome_end = None, length = None, sequence = None,
                 seqtype = 'sequence', source = None, no_homology = False, no_elements = False) -> None:
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
                length (int): Length of the sequence.
                sequence (str): The sequence itself.

                seqtype (str): Description of sequence type, e.g. `sequence` (default) or `exon`
                source (any): Optional reference to where the sequence comes from, useful for genomic elements. MUST be
                                serializable to JSON via `json.dump()`, a string is probably a good idea.
                no_homology (bool): If True, no homology relation is stored. Use this for genomic elements of the 
                                      reference genome.
                no_elements (bool): If True, no genomic elements are stored. Use this for genomic elements.
        """

        assert strand in ['+', '-'], "[ERROR] >>> Strand must be either '+' or '-'."
        assert genome_start >= 0, "[ERROR] >>> Start position must be a positive integer."
        assert type(chromosome) == str, "[ERROR] >>> Chromosome must be a string to avoid undesired behaviour."
        self.species = species
        self.chromosome = chromosome
        self.strand = strand
        self.genome_start = genome_start
        self.type = seqtype
        self.source = source

        try:
            json.dumps(source)
        except TypeError:
            #print("[WARNING] >>> `source` is not serializable to JSON. This might cause problems when saving the " \
            #      + "sequence to a JSON file.")
            logging.warning("[SequenceRepresentation.Sequence.__init__] >>> `source` is not serializable to JSON. " + \
                            "This might cause problems when saving the sequence to a JSON file.")

        assert not (genome_end is None and length is None and sequence is None), \
            "[ERROR] >>> At least one of the following arguments must be provided: genome_end, length, sequence."
        if genome_end is not None:
            assert genome_end >= genome_start, "[ERROR] >>> End position must be greater than start position."
            self.genome_end = genome_end
            self.length = genome_end - genome_start
            if length is not None:
                assert length == genome_end - genome_start, \
                    "[ERROR] >>> Length must be equal to end position minus start position."
            if sequence is not None:
                self.sequence = sequence
                assert self.length == len(sequence), \
                    "[ERROR] >>> Length must be equal to length of the sequence."
        else:
            if length is not None:
                assert length > 0, "[ERROR] >>> Length must be a positive integer."
                self.length = length
                self.genome_end = genome_start + length
                if sequence is not None:
                    self.sequence = sequence
                    assert self.length == len(sequence), \
                        "[ERROR] >>> Length must be equal to length of the sequence."
            else:
                assert sequence is not None, "[ERROR] >>> Sequence must be a string." # should always be true
                self.length = len(sequence)
                self.genome_end = genome_start + len(sequence)
                self.sequence = sequence

        assert hasattr(self, 'genome_end'), "[ERROR] >>> `genome_end` not set."
        assert hasattr(self, 'length'), "[ERROR] >>> `length` not set."

        self._regenerateID()

        if not hasattr(self, 'sequence'):
            self.sequence = None

        if not no_homology:
            self.homology = []
        if not no_elements:
            self.genomic_elements = [] # store exons etc. later

    def __str__(self) -> str:
        """ String representation of the sequence. """
        rep = f"{self.id} ({self.strand}, length = {self.length:,}, type = {self.type})"
        if self.sequence is not None:
            if self.length > 23:
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
        return self.length
    
    def __eq__(self, __value: object) -> bool:
        """ Check if two sequences are equal. """
        if not isinstance(__value, Sequence):
            return False
        eq = self.id == __value.id and self.species == __value.species and self.chromosome == __value.chromosome \
            and self.genome_start == __value.genome_start and self.genome_end == __value.genome_end \
            and self.length == __value.length and self.strand == __value.strand and self.type == __value.type \
            and self.sequence == __value.sequence
        if hasattr(self, 'homology') and hasattr(__value, 'homology'):
            eq = eq and self.homology == __value.homology
        elif hasattr(self, 'homology') or hasattr(__value, 'homology'):
            eq = False
        if hasattr(self, 'genomic_elements') and hasattr(__value, 'genomic_elements'):
            eq = eq and self.genomic_elements == __value.genomic_elements
        elif hasattr(self, 'genomic_elements') or hasattr(__value, 'genomic_elements'):
            eq = False

        return eq
    
    def _regenerateID(self):
        """ Regenerate the sequence ID. Does not need to be called manually. """
        self.id = f"{self.species}:{self.chromosome}:{self.genome_start:,}-{self.genome_end:,}"
    
    def addElement(self, element):
        """ Add a genomic element to the sequence. """
        _addElementToSequence(element, self)

    def addSubsequenceAsElement(self, start: int, end: int, seqtype: str, strand: str = None, source = None,
                                genomic_positions: bool = False, **kwargs):
        """ Define a subsequence of the sequence and add it as a genomic element. 
            Args:
                start (int): Start position of the subsequence.
                end (int): End position of the subsequence.
                seqtype (str): Description of element type, e.g. `exon`
                strand (str): Strand of the subsequence. Can be either '+' or '-'. If None, the strand of the sequence
                                is used.
                source (any): Optional reference to where the sequence comes from, useful for genomic elements. MUST be
                                serializable to JSON via `json.dump()`, a string is probably a good idea.
                genomic_positions (bool): If True, the positions are interpreted as genomic positions, otherwise as
                                            positions within the sequence.
                **kwargs: Additional arguments for the element, i.e. `no_homology` and `no_elements`
        """
        assert start <= end, "[ERROR] >>> `start` must be less than or equal to `end`."
        if strand is None:
                strand = self.strand

        if genomic_positions:            
            element = Sequence(self.species, self.chromosome, strand, start, end, seqtype=seqtype, source=source,
                               **kwargs)
        else:
            element = Sequence(self.species, self.chromosome, strand, self.genome_start+start, self.genome_start+end, 
                               seqtype=seqtype, source=source, **kwargs)
            
        assert _sequencesOverlap(self, element), "[ERROR] >>> Subsequence must overlap with sequence."

        # add subsequence if applicable
        if self.sequence is not None:
            subseqstart = max(0, element.genome_start - self.genome_start)
            subseqend = min(element.genome_end - self.genome_start, len(self.sequence))
            element.sequence = self.sequence[subseqstart:subseqend]

        self.addElement(element)

    def addHomology(self, homology):
        """ Add a homology to the sequence. """
        _addHomologyToSequence(homology, self)

    def elementsPossible(self) -> bool:
        """ Check if the sequence can contain genomic elements. """
        return hasattr(self, 'genomic_elements')
    
    def getElement(self, index):
        """ Get a genomic element of the sequence. """
        assert self.hasElements(), "[ERROR] >>> Sequence does not contain genomic elements."
        assert index < len(self.genomic_elements), "[ERROR] >>> Index out of range."
        return self.genomic_elements[index]

    def getRelativePositions(self, parent, from_rc: bool = False) -> tuple[int, int]:
        """ Get the relative positions of the sequence within a parent sequence as a tuple (start, stop). 
            If `from_rc` is True, the positions are calculated from the reverse complement of the parent sequence. 
            For example: The parent sequence is AAAAAGGGAA and this sequence is GGG. The relative positions are (5, 8).
                         If `from_rc` is True, the relative positions are (2, 5). """
        return _getRelativePositions(self, parent, from_rc)

    def getSequence(self, rc: bool = False) -> str:
        """ Get the sequence of the sequence object. Returns None if no sequence is stored. If `rc` is True, the
            reverse complement of the sequence is returned. """
        if self.sequence is None:
            return None
        elif rc:
            return str(Seq(self.sequence).reverse_complement())
        else:
            return self.sequence
        
    def getSubsequence(self, genome_start, genome_end, rc: bool = False) -> str:
        """ Get a subsequence of the sequence object. Returns None if no sequence is stored or if the requested 
            positions are not in the range of this sequence. If `rc` is True, the reverse complement of the subsequence
            is returned. """
        if self.sequence is None:
            return None
        else:
            seq = self.getSequence()
            if genome_end <= self.genome_start:
                return None
            if genome_start >= self.genome_end:
                return None
            
            start = max(0, genome_start - self.genome_start)
            end = min(genome_end - self.genome_start, len(self.sequence))
            subseq = seq[start:end]
            if rc:
                return str(Seq(subseq).reverse_complement())
            else:
                return subseq
        
    def hasElements(self) -> bool:
        """ Check if the sequence contains genomic elements. """
        return self.elementsPossible() and len(self.genomic_elements) > 0
    
    def hasHomologies(self) -> bool:
        """ Check if the sequence contains homologies. """
        return self.homologiesPossible() and len(self.homology) > 0

    def homologiesPossible(self) -> bool:
        """ Check if the sequence can contain homologies. """
        return hasattr(self, 'homology')

    def isSubsequenceOf(self, parent) -> bool:
        """ Check if the sequence is a subsequence of another Sequence object, i.e. start and end both lie inside the 
            parent sequence. """
        return _sequencesOverlap(self, parent) and self.genome_start >= parent.genome_start \
            and self.genome_end <= parent.genome_end

    def stripSequence(self, amount, from_start = True):
        """ Remove `amount` positions from the sequence object. Discards genomic elements that no longer overlap 
            afterwards, has no impact on homologies.
            Args: 
                amount (int): Number of positions to remove.
                from_start (bool): If True, remove positions from the start of the sequence, otherwise from the end.
            """
        assert amount <= self.length, "[ERROR] >>> `amount` must be less than or equal to the length of the sequence."
        if from_start:
            self.genome_start += amount
            self.length -= amount
            if self.sequence is not None:
                self.sequence = self.sequence[amount:]

        else:
            self.genome_end -= amount
            self.length -= amount
            if self.sequence is not None:
                self.sequence = self.sequence[:-amount]

        if hasattr(self, 'genomic_elements'):
            new_elements = []
            for element in self.genomic_elements:
                if _sequencesOverlap(self, element):
                    new_elements.append(element)

            self.genomic_elements = new_elements

        self._regenerateID() # make sure the new range is reflected in the ID

    def toDict(self) -> dict:
        """ Return a dictionary representation of the sequence. """
        #print("[DEBUG] >>> called .toDict from Sequence", str(self))
        objdict = {
            'id': self.id,
            'species': self.species,
            'chromosome': self.chromosome,
            'strand': self.strand,
            'genome_start': self.genome_start,
            'genome_end': self.genome_end,
            'length': self.length
        }
        if self.sequence is not None:
            objdict['sequence'] = self.sequence

        objdict['type'] = self.type
        if self.source is not None:
            objdict['source'] = self.source

        if hasattr(self, 'genomic_elements'):
            objdict['genomic_elements'] = [element.toDict() for element in self.genomic_elements]
        if hasattr(self, 'homology'):
            objdict['homology'] = [homology.toDict() for homology in self.homology]

        #print("[DEBUG] >>> finished .toDict from Sequence", str(self))
        return objdict
    
    def toTuple(self) -> tuple:
        """ Return a hashable tuple representation of the sequence. The element order is: (id, species, chromosome,
              strand, genome_start, genome_end, length, sequence, type, source, genomic_elements, homologies). Sequence
              is None if no sequence is present, genomic_elements and homologies are None if they are not allowed and
              empty tuples if they are not present. Source is None if no source is present and a json-representation
              from `json.dumps()` if it is present.
        """
        ge = () if self.elementsPossible() else None
        hom = () if self.homologiesPossible else None
        if hasattr(self, 'genomic_elements'):
            ge = tuple([element.toTuple() for element in self.genomic_elements])
        if hasattr(self, 'homology'):
            hom = tuple([homology.toTuple() for homology in self.homology])
        source = None if self.source is None else json.dumps(self.source)

        return (self.id, self.species, self.chromosome, self.strand, self.genome_start, self.genome_end, self.length, 
                self.sequence, self.type, source, ge, hom)
        


# helper functions for adding genomic elements and homology to a sequence
def _addElementToSequence(element: Sequence, sequence: Sequence):
    """ Helper function for adding genomic elements to a sequence, do not use directly """
    #assert type(element) == Sequence, f"[ERROR] >>> `element` must be of type Sequence, not {type(element)}."
    #assert type(sequence) == Sequence, f"[ERROR] >>> `sequence` must be of type Sequence, not {type(sequence)}."
    assert hasattr(sequence, 'genomic_elements'), "[ERROR] >>> `sequence` must have attribute `genomic_elements`."
    #assert element.genome_start >= sequence.genome_start, "[ERROR] >>> `element` must start after `sequence`."
    #assert element.genome_end <= sequence.genome_end, "[ERROR] >>> `element` must end before `sequence`."
    assert _sequencesOverlap(element, sequence), "[ERROR] >>> `element` must overlap with `sequence`."
    sequence.genomic_elements.append(element)



def _addHomologyToSequence(homology: Sequence, sequence: Sequence):
    """ Helper function for adding a homology to a sequence, do not use directly """
    #assert type(homology) == Sequence, f"[ERROR] >>> `homology` must be of type Sequence, not {type(homology)}."
    #assert type(sequence) == Sequence, f"[ERROR] >>> `sequence` must be of type Sequence, not {type(sequence)}."
    assert hasattr(sequence, 'homology'), "[ERROR] >>> `sequence` must have attribute `homology`."
    sequence.homology.append(homology)



def _getRelativePositions(sequence: Sequence, parent: Sequence, from_rc: bool = False) -> tuple[int, int]:
    """ Helper function, returns the relative positions of a sequence within a parent sequence """
    #assert isinstance(sequence, Sequence), f"[ERROR] >>> `sequence` must be of type Sequence, not {type(sequence)}."
    #assert isinstance(parent, Sequence), f"[ERROR] >>> `parent` must be of type Sequence, not {type(parent)}."
    assert _sequencesOverlap(sequence, parent), "[ERROR] >>> `sequence` must overlap with `parent`."

    start = sequence.genome_start - parent.genome_start
    end = sequence.genome_end - parent.genome_start

    if from_rc:
        start_rc = parent.length - end
        end = parent.length - start
        start = start_rc

    return start, end



def _sequencesOverlap(seq1: Sequence, seq2: Sequence) -> bool:
    """ Helper function, returns true if two sequences are from the same species and chromosome and overlap """
    #print("[DEBUG] >>> ", repr(seq1), "\n", repr(seq2))
    return seq1.species == seq2.species and seq1.chromosome == seq2.chromosome and \
        seq1.genome_start < seq2.genome_end and seq1.genome_end > seq2.genome_start



# create a Sequence object from a JSON file or string
def sequenceFromJSON(jsonfile: str = None, jsonstring: str = None) -> Sequence:
    """ Create a Sequence object from a JSON file or a JSON string. 
        If both arguments are provided, the file will be used. """
    assert not (jsonfile is None and jsonstring is None), \
        "[ERROR] >>> Either `jsonfile` or `jsonstring` must be provided."
    if jsonfile is not None:
        assert os.path.isfile(jsonfile), f"[ERROR] >>> file `{jsonfile}` must be a valid file."
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
        length = objdict['length'],
        seqtype = objdict['type'],
        no_elements=not 'genomic_elements' in objdict,
        no_homology=not 'homology' in objdict
    )

    if sequence.id != objdict['id']:
        #print("[WARNING] >>> ID mismatch, setting ID to", objdict['id'])
        #print("              old ID:", sequence.id)
        #print("              You might want to call _regenerateID() on the sequence.")
        logging.warning(f"[SequenceRepresentation.sequenceFromJSON] >>> ID mismatch, setting ID to {objdict['id']}")
        logging.warning(f"                                              old ID: {sequence.id}")
        logging.warning( "                                              You might want to call _regenerateID() on " + \
                        "the sequence.")
        sequence.id = objdict['id']

    if 'sequence' in objdict:
        sequence.sequence = objdict['sequence']
    if 'source' in objdict:
        sequence.source = objdict['source']
    if 'genomic_elements' in objdict:
        for element in objdict['genomic_elements']:
            sequence.addElement(sequenceFromJSON(jsonstring = json.dumps(element)))
    if 'homology' in objdict:
        for homology in objdict['homology']:
            sequence.addHomology(sequenceFromJSON(jsonstring = json.dumps(homology)))

    return sequence



def loadJSONSequenceList(jsonfile: str) -> list[Sequence]:
    """ Load a list of Sequence objects from a JSON file. """
    assert os.path.isfile(jsonfile), f"[ERROR] >>> file `{jsonfile}` must be a valid file."
    with open(jsonfile, 'rt') as f:
        objlist = json.load(f)

    return [sequenceFromJSON(jsonstring = json.dumps(objdict)) for objdict in objlist]



# Class to represent a genome, which is essentially just a list of Sequence objects
class Genome:
    """ Class to represent a genome, which is essentially a blown-up list of Sequence objects. 
        However, this class also provides some useful methods for working with genomes, like querying all sequences
        from the same chromosome (scaffold, ...). It further enforces that the Sequences all belong to the same species.

        Attributes:
            sequences (list[Sequence]): A list of Sequence objects.
            species (str): The species of the genome.
    """	

    def __init__(self, sequences: list[Sequence] = None):
        assert sequences is None or type(sequences) == list, "[ERROR] >>> `sequences` must be None or a list of " \
                                                                           + f"Sequence objects, not {type(sequences)}."
        self.sequences = []
        self.species = None
        self._chromap = {} # chromosome map, maps chromosome names to indices in self.sequences
        if sequences is not None:
            for sequence in sequences:
                self.addSequence(sequence)

    # https://docs.python.org/3/reference/datamodel.html#emulating-container-types
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, key) -> Union[list[Sequence], Sequence]:
        """ Return the chromosome (i.e. list of Sequence objects) with the given key (if key is a chromosome name)
            or the respective Sequence object if key is an array ID."""
        def tryCastToInt(key):
            try:
                return int(key)
            except ValueError:
                return key
            
        # if key is a string, return the respective chromosome (i.e. a list of Sequences)
        if type(key) == str:
            if key in self._chromap:
                # [DEPRECATED] for sequence in self.sequences:
                #    if sequence.chromosome == key:
                #        return sequence
                return [self.sequences[i] for i in self._chromap[key]]
            else:
                raise KeyError(f"Chromosome {key} not found in genome.")
        # int or int-like key -> return the respective Sequence object at the given index
        elif type(key) == int or type(tryCastToInt(key)) == int:
            return self.sequences[key]
        else:
            raise TypeError(f"Invalid key type {type(key)}.")
        
    def __iter__(self):
        return iter(self.sequences)
    
    def __reversed__(self) -> list[Sequence]:
        return reversed(self.sequences)
    
    def __contains__(self, seq_or_chr: Union[Sequence, str]) -> bool:
        """ Return True if the genome contains the given sequence or chromosome name. """
        if type(seq_or_chr) == Sequence:
            return seq_or_chr in self.sequences
        elif type(seq_or_chr) == str:
            return seq_or_chr in self._chromap
        else:
            raise KeyError("[ERROR] >>> `sequence` must be of type Sequence or str (chromosome name).")
        
    def __str__(self) -> str:
        return f"Genome from {self.species} with {len(self._chromap)} chromosomes: " \
            + f"{', '.join([f'{chrom} ({len(self[chrom])} sequence[s])' for chrom in self._chromap])}"
    
    def __repr__(self) -> str:
        return str(self)
    
    def addSequence(self, sequence: Sequence):
        """ Adds a Sequence to the genome and performs checks to ensure that the genome is valid. """
        #assert type(sequence) == Sequence, f"[ERROR] >>> `sequence` must be of type Sequence, not {type(sequence)}."
        if self.species is None:
            self.species = sequence.species
        else:
            assert self.species == sequence.species, \
                f"[ERROR] >>> All sequences in a genome must be from the same species {self.species}. " \
                + f" You tried to add a sequence from {sequence.species}."
        # [DEPRECATED] assert sequence.chromosome not in [seq.chromosome for seq in self.sequences], \
        #    f"[ERROR] >>> All chromosomes in a genome must be unique, {sequence.chromosome} is already in the genome."
        
        if sequence.chromosome not in self._chromap:
            self._chromap[sequence.chromosome] = []

        self._chromap[sequence.chromosome].append(len(self.sequences))
        self.sequences.append(sequence)

    def getSequenceStrings(self) -> list[str]:
        """ Returns a list of all sequences strings in the genome. """
        return [sequence.sequence for sequence in self.sequences]
    
    def toDict(self) -> list[dict]:
        """ Returns a list of dictionary representations of the sequences in the genome. """
        return [sequence.toDict() for sequence in self.sequences]
    
    def toJSON(self, file: str):
        """ Write the genome to a JSON file. """
        with open(file, 'wt') as f:
            json.dump(self.toDict(), f, indent=4)



def genomeFromJSON(jsonfile: str = None, jsonstring: str = None) -> Genome:
    """ Create a Genome object from a JSON file or a JSON string.
        If both arguments are provided, the file will be used. """
    assert not (jsonfile is None and jsonstring is None), \
        "[ERROR] >>> Either `jsonfile` or `jsonstring` must be provided."
    if jsonfile is not None:
        assert os.path.isfile(jsonfile), f"[ERROR] >>> file `{jsonfile}` must be a valid file."
        with open(jsonfile, 'rt') as f:
            dictlist = json.load(f)
    else:
        dictlist = json.loads(jsonstring)

    return Genome(sequences = [sequenceFromJSON(jsonstring = json.dumps(objdict)) for objdict in dictlist])
    


def loadJSONGenomeList(jsonfile: str) -> list[Genome]:
    """ Load a list of Genome objects from a JSON file. """
    assert os.path.isfile(jsonfile), f"[ERROR] >>> file `{jsonfile}` must be a valid file."
    with open(jsonfile, 'rt') as f:
        objlist = json.load(f)

    return [genomeFromJSON(jsonstring = json.dumps(objdict)) for objdict in objlist]