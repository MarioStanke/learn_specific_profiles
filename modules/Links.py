from dataclasses import dataclass
import json
import logging
import itertools

from . import SequenceRepresentation as sr
from . import utils
from .typecheck import typecheck, typecheck_list, typecheck_objdict, typecheck_objdict_list

@dataclass
class Occurrence:
    """ Coordinates of a position in a genomic (DNA) sequence. Attribute position should always refer to the top strand,
     even if the referred-to genomic element is on the bottom strand. Such information is stored in the strand 
     attribute (strand can be either '+' or '-'). `sitelen` must be at least 1, `profileIdx` must be >= 0 or -1 if no
     profile is associated with the occurrence. """
    sequence: sr.Sequence
    position: int
    strand: str
    sitelen: int = 1
    profileIdx: int = -1 # -1 for "no profile"

    def __post_init__(self):
        assert typecheck(self.sequence, "Sequence", die=True), \
            "[ERROR] >>> Invalid sequence type: " + str(type(self.sequence))
        assert self.position >= 0, "[ERROR] >>> Negative position: " + str(self.position)
        assert self.strand in ['+', '-'], "[ERROR] >>> Invalid strand: " + self.strand
        assert self.sitelen > 0, "[ERROR] >>> Invalid site length: " + str(self.sitelen)
        assert self.profileIdx >= -1 , "[ERROR] >>> Invalid profile index: " + str(self.profileIdx)

        self.classname = "Occurrence" # for typechecking

    def __str__(self) -> str:
        return(f"{self.sequence.id}\tp = {self.position}\t{self.strand}\tu = {self.profileIdx}")
    def __repr__(self) -> str:
        return self.__str__()
    def __lt__(self, other):
        return self.tuple() < other.tuple()
    def __eq__(self, other):
        return self.tuple() == other.tuple()
    def __hash__(self):
        return hash(self.tuple()) # TODO: can probably be deprecated in the next big refactoring when Occurrences hold refs to Sequences
    
    def getSite(self) -> str:
        """ Returns the site that the Occurrence refers to. If strand is '-', it takes the 
            _top strand_ sequence slice [position:position+sitelen] and returns the reverse complement of that slice."""
        if self.strand == '+':
            site = self.sequence.getSlice(self.position, self.position+self.sitelen)
        else:
            site = self.sequence.getSlice(self.position, self.position+self.sitelen, rc=True)

        assert site is not None and len(site) == self.sitelen, \
            f"[ERROR] >>> Invalid site or length: {site=}, {self.sitelen=}"
        return site        
    
    def tuple(self):
        """ Attention: does not include the sequence reference, only the ID! """
        return (self.sequence.id, self.position, self.strand, self.sitelen, self.profileIdx)
    
    def toDict(self):
        """ Returns a dictionary representation of the Occurrence, with the complete dictionary of the Sequence. """
        return {"sequence": self.sequence.toDict(), 
                "position": self.position, 
                "strand": self.strand, 
                "sitelen": self.sitelen,
                "profileIdx": self.profileIdx,
                "classname": self.classname}
    

def occurrenceFromTuple(t: tuple, seq_collection, die: bool = True) -> Occurrence:
    """ Creates an Occurrence from a tuple that was created via Occurrence.tuple() and the corresponding 
    Sequence object. 
    `seq_collection` can be a Sequence, a Genome, or a list (or list of lists) of Sequences or Genomes. It _must_
    contain the Sequence object that corresponds to the Occurrence's sequence ID. If `die` is False, None is returned
    if the Sequence object is not found.
    
    Args:
        t (tuple): tuple representation of the Occurrence
        seq_collection: Sequence, Genome, or list of Sequences or Genomes
        die (bool, optional): If True, raise an error if the Sequence object is not found. Defaults to True.
        
    Returns:
        Occurrence: The Occurrence object

    Raises:
        AssertionError: If the Sequence object's ID does not match the ID in the tuple and `die` is True.
        ValueError: If the Sequence object is not found in `seq_collection` and `die` is True, 
                    or if `seq_collection` is not a valid type.
    """
    if typecheck(seq_collection, "Sequence", die=False, log_warnings=False):
        sequence = seq_collection
        seqmatch = sequence.id == t[0]
        if seqmatch:
            return Occurrence(sequence, t[1], t[2], t[3], t[4])
        elif not die:
            return None
        else:
            raise AssertionError(f"[ERROR] >>> Sequence ID mismatch: {sequence.id} != {t[0]}")
    
    elif typecheck(seq_collection, "Genome", die=False, log_warnings=False):
        genome = seq_collection
        for sequence in genome:
            if sequence.id == t[0]:
                return Occurrence(sequence, t[1], t[2], t[3], t[4])
            
        if not die:
            return None
        else:
            raise ValueError(f"[ERROR] >>> Sequence ID {t[0]} not found in genome {genome}.")
        
    elif type(seq_collection) is list:
        for element in seq_collection:
            occ = occurrenceFromTuple(t, element, die=False)
            if occ is not None:
                break

        if occ is not None:
            return occ
        elif not die:
            return None
        else:
            raise ValueError(f"[ERROR] >>> Sequence ID {t[0]} not found in list {seq_collection}.")

    else: # always die
        raise ValueError(f"[ERROR] >>> Invalid seq_collection type: {type(seq_collection)}")



def occurrenceFromDict(d: dict) -> Occurrence:
    """ Creates an Occurrence from a dictionary that was created via Occurrence.toDict(). """
    assert typecheck_objdict(d, "Occurrence", die=True)
    sequence = sr.sequenceFromJSON(jsonstring=json.dumps(d["sequence"]))
    return Occurrence(sequence, d["position"], d["strand"], d['sitelen'], d["profileIdx"])



# Class representing MultiLinks, i.e. multple occurrences per genome are allowed
class MultiLink:
    """ A generalization of the Link class that allows multiple occurrences per genome. 
    Attributes:
        occs: list of lists of Occurrences, one list per genome that has occurrences
    """

    classname = "MultiLink"

    def __init__(self, occs: list[Occurrence], singleProfile: bool = True):
        """ Constructor for MultiLink.
        Args:
            occs: list of Occurrences
            singleProfile: bool, default True. Indicates if all Occurrences come from the same profile.
        """
        self.occs: list[list[Occurrence]] = []
        self._singleProfile = singleProfile
        self._speciesToOccListIdx: dict[str, int] = {}

        if self._singleProfile:
            assert len(set([occ.profileIdx for occ in occs])) == 1, "[ERROR] >>> Single-profile MultiLink, " \
                                         + f"but multiple profile indices: {set([occ.profileIdx for occ in occs])}"
        occs = sorted(occs)
        for occ in occs:
            assert occ.position >= 0, f"[ERROR] >>> Negative position: {occ.position} (sequence {occ.sequence.id})"
            assert occ.position < len(occ.sequence), f"[ERROR] >>> Position out of range (0, {len(occ.sequence)}): " \
                                                      + f"{occ.position} (sequence {occ.sequence.id})"

            if occ.sequence.species not in self._speciesToOccListIdx:
                self._speciesToOccListIdx[occ.sequence.species] = len(self.occs)
                self.occs.append([])
                
            self.occs[self._speciesToOccListIdx[occ.sequence.species]].append(occ) # add occ to list of occs for genome

    def __str__(self):
        s = []
        for occs in self.occs:
            for occ in occs:
                s.append(f"{occ.tuple()}")

        return '['+"\n ".join(s)+']'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        """ Returns number of genomes that have occurrences in the MultiLink. """
        return len(self.occs)
    
    def __iter__(self):
        return iter(self.occs)

    def getSpeciesOccurrences(self, species: str) -> list[Occurrence] | None:
        """ Returns the list of occurrences that belong to species, or None if species is not found. """
        for occs in self.occs:
            if occs[0].sequence.species == species:
                return occs
            
        return None
    
    def nUniqueLinks(self):
        """ Returns the number of unique links that can be created from the MultiLink """
        if self._singleProfile:
            # attention: np.prod() does not handle numeric overflow! However, there is no integer overflow in Python
            nlinks = 1
            for occs in self.occs:                
                nlinks *= len(occs)
                
            return nlinks
        else:
            return _calculateUniqueLinksFromMultiprofileMultiLink(self)

    def singleProfile(self) -> bool:
        return self._singleProfile
    
    def size(self) -> tuple[int, int]:
        """ Returns a tuple with the first element being the number of genomes 
        and the second the total number of occurrences. """
        return (len(self.occs), sum([len(occs) for occs in self.occs]))
    
    def toDict(self):
        """ Returns a dictionary representation of the Link. Important: Does _not_ contain the whole genomes, but only
         the genome IDs in the occurrences. """
        return {"occs": [[list(occ.tuple()) for occ in occs] for occs in self.occs], 
                "singleProfile": self._singleProfile,
                "classname": self.classname}

    def toLinks(self, linkThreshold: int = 100) -> list | None:
        """ Converts the MultiLink to a list of Links. If the number of Links would exceed linkThreshold,
            None is returned. """
        lfor = linksFromOccurrences([occ for occs in self.occs for occ in occs], linkThreshold)
        links = lfor.links
        skipped = lfor.skipped
        if len(skipped) > 0:
            logging.warning(f"[MultiLink.toLinks] >>> Skipped {len(skipped)} profiles: {skipped}")
            if self._singleProfile:
                assert len(links) == 0, "[ERROR] >>> Skipped profiles, but links were created."
                return None
            else:
                if len(links) > 0:
                    logging.warning(f"[MultiLink.toLinks] >>> Not all possible links were created.")
                    return links
                else:
                    return None
        else:
            assert len(links) > 0, "[ERROR] >>> No links created."
            return links



def _calculateUniqueLinksFromMultiprofileMultiLink(link: MultiLink) -> int:
    """ Calculates the number of unique links that can be created from a MultiLink with multiple profiles. """
    profileToOccs: dict[int, list[Occurrence]] = {}
    for occs in link.occs:
        for occ in occs:
            if occ.profileIdx not in profileToOccs:
                profileToOccs[occ.profileIdx] = []
            profileToOccs[occ.profileIdx].append(occ)

    nUniqueLinks = 0
    for profileidx in profileToOccs:
        # very important that singleProfile is set to true, otherwise infinite recursion!
        uniqueMultiLink = MultiLink(profileToOccs[profileidx], singleProfile=True)
        nUniqueLinks += uniqueMultiLink.nUniqueLinks()
        
    return nUniqueLinks	



# Class representing Links
@dataclass
class Link:
    occs: list[Occurrence]

    # needed for X-Drop-like expansion
    expandMatchParameter: float = 5
    expandMismatchParameter: float = -4
    expandScoreThreshold: float = 20
    expandX: float = 100

    # control check for single-profile links
    _singleProfile: bool = True

    def __post_init__(self):
        self.classname = "Link"

        self.occs = sorted(self.occs)
        genoccs = [o.sequence.species for o in self.occs]
        assert genoccs == sorted(set(genoccs)), "[ERROR] >>> Genomes are not unique: " + str(genoccs)
        if self._singleProfile:
            assert len(set([occ.profileIdx for occ in self.occs])) == 1, "[ERROR] >>> Single-profile Link, " \
                                         + f"but multiple profile indices: {set([occ.profileIdx for occ in self.occs])}"

        for occ in self.occs:
            assert occ.position < len(occ.sequence), "[ERROR] >>> Position index out of range: " + str(occ.position) \
                                                        + f"for {occ=}"

    def __str__(self):
        for occ in self.occs:
            print(occ)

    def __len__(self):
        return len(self.occs)
    
    def __getitem__(self, key):
        return self.occs[key]
    
    def __iter__(self):
        return iter(self.occs)

    def getSpeciesOccurrence(self, species: str) -> Occurrence | None:
        """ Returns the occurrence that belongs to species, or None if species is not found. """
        for occ in self.occs:
            if occ.sequence.species == species:
                return occ
            
        return None
    
    def MSA(self, aa = False):
        """ Prints a multiple sequence alignment of the sequences in the link.
            If aa is True, the sequences are translated to amino acids before alignment. """
        for occ in self.occs:
            seq = occ.sequence.getSequence()[occ.position:occ.position+occ.sitelen]
            if occ.strand == '-':
                seq = sr.Sequence("", "", "-", 0, sequence=seq).getSequence(rc = True)
                
            if aa:
                seq = utils.sequence_translation(seq)

            print(seq, "-", occ.sequence.id, f"at {occ.position}:{occ.position+occ.sitelen}")
    
    def expand(self, matchParameter: float = None, mismatchParameter: float = None, 
               scoreThreshold: float = None, x: float = None):
        """ Link X-Drop-like expansion 
        
        Args:
            matchParameter (float, optional): Match parameter. Defaults to None, in this case the value of 
                                                self.expandMatchParameter is used.
            mismatchParameter (float, optional): Mismatch parameter. Defaults to None, in this case the value of
                                                    self.expandMismatchParameter is used.
            scoreThreshold (float, optional): Score threshold. Defaults to None, in this case the value of
                                                self.expandScoreThreshold is used.
            x (float, optional): X parameter. Defaults to None, in this case the value of self.expandX is used.
        """
        pass

    def singleProfile(self) -> bool:
        return self._singleProfile

    def toDict(self):
        """ Returns a dictionary representation of the Link. Important: Does _not_ contain the whole genomes, but only
         the genome IDs in the occurrences. """
        return {"occs": [list(occ.tuple()) for occ in self.occs], 
                "expandMatchParameter": self.expandMatchParameter,
                "expandMismatchParameter": self.expandMismatchParameter,
                "expandScoreThreshold": self.expandScoreThreshold,
                "expandX": self.expandX,
                "singleProfile": self._singleProfile,
                "classname": self.classname}
    
    def toMultiLink(self) -> MultiLink:
        return MultiLink(self.occs, singleProfile=True)



def linkFromDict(d: dict, genomes: list[sr.Genome]) -> Link | MultiLink:
    """ Creates a Link from a dictionary representation that was created via the toDict() member. Returns either a 
     Link or a MultiLink, depending on the inputs `classname` tag. """
    assert typecheck_objdict_list(d, ["Link", "MultiLink"], die=True)
    
    if d["classname"] == "Link":
        occs = [occurrenceFromTuple(occ, genomes) for occ in d["occs"]]
        return Link(occs, d["expandMatchParameter"], d["expandMismatchParameter"],
                    d["expandScoreThreshold"], d["expandX"], d["singleProfile"])
    else:
        occs = [occurrenceFromTuple(occ, genomes) for occs in d["occs"] for occ in occs]
        return MultiLink(occs, d["singleProfile"])


def linkFromDict_fast(d: dict, s: list[sr.Sequence | list[sr.Sequence]]) -> Link | MultiLink:
    """ Creates a Link from a dictionary representation that was created via the toDict() member. Returns either a 
     Link or a MultiLink, depending on the inputs `classname` tag. `occs` has to be the same 'shape' as d['occs'],
     containing the correct Sequence object for each occurrence tuple. """
    assert typecheck_objdict_list(d, ["Link", "MultiLink"], die=True)
    assert len(s) == len(d['occs']), f"[ERROR] >>> Occurrence list length mismatch: {len(s)} != {len(d['occs'])}"
    if d["classname"] == "Link":
        assert all([s[i].id() == d['occs'][i][0] for i in range(len(s))]), \
            f"[ERROR] >>> Occurrence Sequence mismatch in Link: {[seq.id for seq in s]} != {d['occs']}"
    else:
        assert all(len(s[i]) == len(d['occs'][i]) for i in range(len(s))), "[ERROR] >>> Occurrence list length " \
            + f"mismatch in MultiLink: {[seq.id for ss in s for seq in ss]} != {d['occs']}"
        assert all([s[i][j].id == d['occs'][i][j][0] for i in range(len(s)) for j in range(len(s[i]))]), \
           f"[ERROR] >>> Occurrence Sequence mismatch in MultiLink: {[seq.id for ss in s for seq in ss]} != {d['occs']}"
        
    if d["classname"] == "Link":
        occs = [occurrenceFromTuple(occ, s[i]) for i, occ in enumerate(d["occs"])]
        return Link(occs, d["expandMatchParameter"], d["expandMismatchParameter"],
                    d["expandScoreThreshold"], d["expandX"], d["singleProfile"])
    else:
        occs = [occurrenceFromTuple(occ, s[i][j]) for i, occslist in enumerate(d["occs"]) \
                    for j, occ in enumerate(occslist)] # flattened list for MutliLink construcor
        return MultiLink(occs, d["singleProfile"])



def nLinks(links: list[Link | MultiLink]) -> int:
    """ Returns the number of unique links in a list of Links or MultiLinks. """
    nlinks = 0
    for l in links:
        typecheck_list(l, ["Link", "MultiLink"], die=True)
        if l.classname == "MultiLink":
            nlinks += l.nUniqueLinks()
        else:
            nlinks += 1
    
    return nlinks



@dataclass
class linksFromOccurrencesResult:
    links: list[Link]
    skipped: list[tuple[int, int]]


def linksFromOccurrences(occurences: list[Occurrence], linkThreshold: int = 100) -> linksFromOccurrencesResult:
    """ Creates links from a list of occurrences. """
    # sort occurrences by profile index and genome index
    profileToGenomeToOccIdx = {}
    for oIdx, occ in enumerate(occurences):
        if occ.profileIdx not in profileToGenomeToOccIdx:
            profileToGenomeToOccIdx[occ.profileIdx] = {}
        if occ.sequence.species not in profileToGenomeToOccIdx[occ.profileIdx]:
            profileToGenomeToOccIdx[occ.profileIdx][occ.sequence.species] = []

        profileToGenomeToOccIdx[occ.profileIdx][occ.sequence.species].append(oIdx)

    # make single-profile links, i.e. only one occurrence per genome in each link. Skip if link threshold is exceeded.
    links: list[Link] = []
    skipped: list[tuple[int, int]] = []
    for pIdx in profileToGenomeToOccIdx:
        if (len(profileToGenomeToOccIdx[pIdx].keys()) == 1):
            continue # no link possible, only one genome has occurrences
            
        occIdcs: list[list[int]] = [profileToGenomeToOccIdx[pIdx][g] for g in profileToGenomeToOccIdx[pIdx]]
            
        nlinks = 1
        for og in occIdcs:
            nlinks *= len(og)
            if nlinks > linkThreshold:
                break

        if nlinks > linkThreshold:
            logging.debug(f"[Links.linksFromOccurrences] >>> Profile {pIdx} would produce at least {nlinks} links, " \
                          + "skipping")
            skipped.append((pIdx, nlinks))
        else:
            idxLinks: list[list[int]] = list(itertools.product(*occIdcs))
            for l in idxLinks:
                loccs = [occurences[i] for i in l]
                links.append(Link(loccs))

    return linksFromOccurrencesResult(links, skipped)



def linksFromMultiLinks(mlinks: list[MultiLink], link_threshold: int = 100) -> list[Link]:
    """ Creates a list of Links from a list of MultiLinks. If a MultiLink would create more than link_threshold links,
     it is skipped. """
    links = []
    for mlink in mlinks:
        typecheck(mlink, "MultiLink", die=True)
        ln = mlink.toLinks(link_threshold)
        if ln is not None:
            links.extend(ln)
    
    return links



def multiLinksFromOccurrences(occurrences: list[Occurrence], singleProfileLinks = True) -> list[MultiLink]:
    """ Creates a list of MultiLinks from a list of Occurrences. If `singleProfileLinks` is True (default), each
     MultiLink is created only from Occurrences with the same profile index. Otherwise it's basically just calling
     MultiLink(occurrences)."""
    if singleProfileLinks:
        profileToOccs = {}
        for occ in occurrences:
            if occ.profileIdx not in profileToOccs:
                profileToOccs[occ.profileIdx] = []
            profileToOccs[occ.profileIdx].append(occ)
        
        return [MultiLink(profileToOccs[p], True) for p in profileToOccs]
    else:
        singleProfile = len(set([o.profileIdx for o in occurrences])) == 1
        return [MultiLink(occurrences, singleProfile)]