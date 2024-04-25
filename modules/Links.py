from dataclasses import dataclass, field
import logging
import itertools
import numpy as np

#import model
from . import SequenceRepresentation
from . import sequtils as su

@dataclass
class Occurrence:
    """ Coordinates of a position in a genomic (DNA) sequence. Attribute position should always refer to the top strand,
     even if the referred-to genomic element is on the bottom strand. Such information is stored in the strand 
     attribute (strand can be either '+' or '-')."""
    genomeIdx: int
    sequenceIdx: int
    position: int
    strand: str
    profileIdx: int

    def __post_init__(self):
        assert self.strand in ['+', '-'], "[ERROR] >>> Invalid strand: " + self.strand

    def __str__(self) -> str:
        #return(f"(g: {self.genomeIdx}, c: {self.sequenceIdx}, p: {self.position}, f: {self.frame}, "\
        #       + f"u: {self.profileIdx})")
        return(f"(g: {self.genomeIdx}, c: {self.sequenceIdx}, p: {self.position}, s: {self.strand}, "\
               + f"u: {self.profileIdx})")
    def __repr__(self) -> str:
        return self.__str__()
        #return (f"Occurrence(genomeID={self.genomeIdx}, sequenceID={self.sequenceIdx}, position={self.position}, " \
        #        +f"frame={self.frame}, profileID={self.profileIdx})")
    def __lt__(self, other):
        return self.tuple() < other.tuple()
    def __eq__(self, other):
        return self.tuple() == other.tuple()
    def __hash__(self):
        return hash(self.tuple()) # can probably be deprecated in the next big refactoring when Occurrences hold refs to Sequences
    def tuple(self):
        return (self.genomeIdx, self.sequenceIdx, self.position, self.strand, self.profileIdx)



# Class representing MultiLinks, i.e. multple occurrences per genome are allowed
class MultiLink:
    """ A generalization of the Link class that allows multiple occurrences per genome. 
    Attributes:
        occs: list of lists of Occurrences, one list per genome that has occurrences (i.e.: not necessarily the same
              length as `genomes`!)
        span: int
        genomes: list of SequenceRepresentation.Genome
    """

    classname = "MultiLink"

    def __init__(self, occs: list[Occurrence], span: int, genomes: list[SequenceRepresentation.Genome],
                 singleProfile: bool = True):
        """ Constructor for MultiLink.
        Args:
            occs: list of Occurrences
            span: int
            genomes: list of SequenceRepresentation.Genome
            singleProfile: bool, default True. Indicates if all Occurrences come from the same profile.
        """
        self.occs = []
        self.span = span
        self.genomes = genomes
        self.singleProfile = singleProfile
        self._genomeIdxToOccListIdx: dict[int, list[int]] = {}

        assert self.span > 0, "[ERROR] >>> Span must be positive: " + str(self.span)
        if self.singleProfile:
            assert len(set([occ.profileIdx for occ in occs])) == 1, "[ERROR] >>> Single-profile MultiLink, " \
                                         + f"but multiple profile indices: {set([occ.profileIdx for occ in occs])}"
        occs = sorted(occs)
        for occ in occs:
            # print("[DEBUG] >>> occ:", occ)
            assert occ.genomeIdx < len(self.genomes), "[ERROR] >>> Genome index out of range: " + str(occ.genomeIdx)
            assert occ.sequenceIdx < len(self.genomes[occ.genomeIdx]), \
                "[ERROR] >>> Sequence index out of range: " + str(occ.sequenceIdx)
            assert occ.position >= 0, f"[ERROR] >>> Negative position: {occ.position} " \
                + f"(sequence {self.genomes[occ.genomeIdx][occ.sequenceIdx].id})"
            assert occ.position < len(self.genomes[occ.genomeIdx][occ.sequenceIdx]), \
                f"[ERROR] >>> Position out of range (0, {len(self.genomes[occ.genomeIdx][occ.sequenceIdx])}): " \
                    + f"{occ.position} (sequence {self.genomes[occ.genomeIdx][occ.sequenceIdx].id})"

            if occ.genomeIdx not in self._genomeIdxToOccListIdx:
                self._genomeIdxToOccListIdx[occ.genomeIdx] = len(self.occs)
                self.occs.append([])
                
            self.occs[self._genomeIdxToOccListIdx[occ.genomeIdx]].append(occ) # add occ to list of occs for genome

    def __str__(self):
        s = []
        for occs in self.occs:
            for occ in occs:
                s.append(f"{self.genomes[occ.genomeIdx][occ.sequenceIdx].id}\t{occ.tuple()[2]}")

        return '['+"\n ".join(s)+']'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        """ Returns number of genomes that have occurrences in the MultiLink. """
        return len(self.occs)
    
    #def __getitem__(self, key):
    #    return self.occs[key]
    
    def __iter__(self):
        return iter(self.occs)
    
    def nUniqueLinks(self):
        """ Returns the number of unique links that can be created from the MultiLink. """
        if self.singleProfile:
            return int(np.prod([len(occs) for occs in self.occs])) # json will not like np.int64
        else:
            return _calculateUniqueLinksFromMultiprofileMultiLink(self)

    def size(self) -> tuple[int, int]:
        """ Returns a tuple with the first element being the number of genomes 
        and the second the total number of occurrences. """
        return (len(self.occs), sum([len(occs) for occs in self.occs]))
    
    def toDict(self):
        """ Returns a dictionary representation of the Link. Important: Does _not_ contain the whole genomes, but only
         a list of the genome IDs (i.e. a list of Sequence IDs per Genome). """
        return {"occs": [[list(occ.tuple()) for occ in occs] for occs in self.occs], 
                "span": self.span, 
                "genomes": [[seq.id for seq in genome] for genome in self.genomes],
                "singleProfile": self.singleProfile,
                "classname": self.classname}

    # TODO: frame information in sites array is useless, but model.get_profile_match_sites needs to be fixed before 
    #       removing it here.
    def toLinks(self, linkThreshold: int = 100) -> list:
        """ Converts the MultiLink to a list of Links. If the number of Links would exceed linkThreshold,
            None is returned. """
        sites = np.array([[o.genomeIdx, o.sequenceIdx, o.position, o.profileIdx, 0 if o.strand == '+' else 3] \
                            for occs in self.occs for o in occs],
                         dtype=np.int32)
        links, _, skipped = linksFromSites(sites, self.span, self.genomes, linkThreshold)
        if len(skipped) > 0:
            logging.warning(f"[MultiLink.toLinks] >>> Skipped {len(skipped)} profiles: {skipped}")
            if self.singleProfile:
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
    profileToOccs = {}
    for occs in link.occs:
        for occ in occs:
            if occ.profileIdx not in profileToOccs:
                profileToOccs[occ.profileIdx] = []
            profileToOccs[occ.profileIdx].append(occ)

    nUniqueLinks = 0
    for profileidx in profileToOccs:
        uniqueMultiLink = MultiLink(profileToOccs[profileidx], link.span, link.genomes, singleProfile=True)
        # very important that singleProfile is set to true, otherwise infinite recursion!
        nUniqueLinks += uniqueMultiLink.nUniqueLinks()
        
    return nUniqueLinks	



# Class representing Links
@dataclass
class Link:
    occs: list[Occurrence]
    span: int
    genomes: list[SequenceRepresentation.Genome]

    # needed for X-Drop-like expansion
    expandMatchParameter: float = 5
    expandMismatchParameter: float = -4
    expandScoreThreshold: float = 20
    expandX: float = 100

    # control check for single-profile links
    singleProfile: bool = True

    def __post_init__(self):
        self.classname = "Link"

        assert self.span > 0, "[ERROR] >>> Span must be positive: " + str(self.span)
        self.occs = sorted(self.occs)
        genoccs = [o.genomeIdx for o in self.occs]
        assert genoccs == sorted(set(genoccs)), "[ERROR] >>> Genomes are not unique: " + str(genoccs)
        if self.singleProfile:
            assert len(set([occ.profileIdx for occ in self.occs])) == 1, "[ERROR] >>> Single-profile Link, " \
                                         + f"but multiple profile indices: {set([occ.profileIdx for occ in self.occs])}"

        for occ in self.occs:
            assert occ.genomeIdx < len(self.genomes), "[ERROR] >>> Genome index out of range: " + str(occ.genomeIdx)
            assert occ.sequenceIdx < len(self.genomes[occ.genomeIdx]), \
                "[ERROR] >>> Sequence index out of range: " + str(occ.sequenceIdx)
            assert occ.position < len(self.genomes[occ.genomeIdx][occ.sequenceIdx]), \
                "[ERROR] >>> Position index out of range: " + str(occ.position)

    def __str__(self):
        for occ in self.occs:
            print(self.genomes[occ.genomeIdx][occ.sequenceIdx].id, end="\t")
            print(occ[2])

    def __len__(self):
        return len(self.occs)
    
    def __getitem__(self, key):
        return self.occs[key]
    
    def __iter__(self):
        return iter(self.occs)
    
    def MSA(self, aa = False):
        """ Prints a multiple sequence alignment of the sequences in the link.
            If aa is True, the sequences are translated to amino acids before alignment. """
        for occ in self.occs:
            seqRep = self.genomes[occ.genomeIdx][occ.sequenceIdx]
            seq = seqRep.getSequence()[occ.position:occ.position+self.span]
            #if occ.frame >= 3:
            if occ.strand == '-':
                seq = SequenceRepresentation.Sequence("", "", "-", 0, sequence=seq).getSequence(rc = True)
                
            if aa:
                seq = su.sequence_translation(seq)

            print(seq, "-", self.genomes[occ.genomeIdx][occ.sequenceIdx].id, 
                  f"at {occ.position}:{occ.position+self.span}")
    
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

    def toDict(self):
        """ Returns a dictionary representation of the Link. Important: Does _not_ contain the whole genomes, but only
         a list of the genome IDs (i.e. a list of Sequence IDs per Genome). """
        return {"occs": [list(occ.tuple()) for occ in self.occs], 
                "span": self.span, 
                "genomes": [[seq.id for seq in genome] for genome in self.genomes],
                "expandMatchParameter": self.expandMatchParameter,
                "expandMismatchParameter": self.expandMismatchParameter,
                "expandScoreThreshold": self.expandScoreThreshold,
                "expandX": self.expandX,
                "singleProfile": self.singleProfile,
                "classname": self.classname}



def linkFromDict(d: dict, genomes: list[SequenceRepresentation.Genome]) -> Link | MultiLink:
    """ Creates a Link from a dictionary representation that was created via the toDict() member. Returns either a 
     Link or a MultiLink, depending on the inputs `classname` tag. """
    assert "classname" in d, "[ERROR] >>> No classname in dictionary representation of Link."
    assert d["classname"] in ["Link", "MultiLink"], \
        f"[ERROR] >>> Invalid classname in dictionary representation of Link: '{d['classname']}'"
    
    if d["classname"] == "Link":
        occs = [Occurrence(*occ) for occ in d["occs"]]
    else:
        occs = [Occurrence(*occ) for occs in d["occs"] for occ in occs]

    # assert correct genomes
    assert len(d["genomes"]) == len(genomes), \
        f"[ERROR] >>> Number of genomes does not match: {len(d['genomes'])} != {len(genomes)}"
    for i in range(len(genomes)):
        assert len(d["genomes"][i]) == len(genomes[i]), \
            f"[ERROR] >>> Number of sequences in genome {i} does not match: {len(d['genomes'][i])} != {len(genomes[i])}"
        for j in range(len(genomes[i])):
            assert d["genomes"][i][j] == genomes[i][j].id, \
                f"[ERROR] >>> Sequence ID {i}:{j} does not match: {d['genomes'][i][j]} != {genomes[i][j].id}"
    
    if d["classname"] == "Link":
        return Link(occs, d["span"], genomes, d["expandMatchParameter"], d["expandMismatchParameter"],
                    d["expandScoreThreshold"], d["expandX"], d["singleProfile"])
    else:
        return MultiLink(occs, d["span"], genomes, d["singleProfile"])



# TODO: Currently expects genomic positions, not AA-positions. Not a problem, but why do we have frames in the input then?
# sites from model.get_profile_match_sites are currently imprecise, should be safe to just ignore the frame. In the future,
# either do the site conversion here or somewhere else and do it properly then.
def linksFromSites(sites: np.ndarray, span: int, genomes: list[SequenceRepresentation.Genome], linkThreshold = 100) \
    -> tuple[list[Link], set[tuple[int, int, str]], list[tuple[int, int]]]:
    """ Creates links from sites 
    
    Args:
        sites (np.ndarray): Array of sites of shape (X, 5) where X is the number of sites and the second dimension is 
                              (genomeIdx, sequenceIdx, position, profileIdx, frame)
        genomes (list[SequenceRepresentation.Genome]): List of genomes
        linkThreshold (int, optional): Threshold for creating links. Profiles that would create more than linkThreshold
                                        links are ignored. Defaults to 100.
    """

    if type(sites) != np.ndarray:
        sites = sites.numpy() # catch tensor

    links = []
    skipped = []
    profileToOcc = {}
    linkProfiles = set()
    for _g, _c, _p, _u, _f in sites:
        g = int(_g) # json will not like np.int32 later, so just convert everything to int
        c = int(_c)
        p = int(_p)
        u = int(_u)
        f = int(_f)
        if u not in profileToOcc:
            profileToOcc[u] = {}
            
        if g not in profileToOcc[u]:
            profileToOcc[u][g] = []
            
        profileToOcc[u][g].append([g,c,p,f,u])
        
    for u in sorted(profileToOcc.keys()):
        if (len(profileToOcc[u].keys()) == 1):
            continue
            
        occs = []
        for g in profileToOcc[u]:
            occs.append(profileToOcc[u][g])
            
        # attention: nlinks = np.prod([len(og) for og in occs]) does not handle numeric overflow!
        nlinks = 1
        for og in occs:
            nlinks *= len(og)
            if nlinks > linkThreshold:
                break

        if nlinks > linkThreshold:
            #print("[DEBUG] >>> Profile", u, "would produce at least", nlinks, "links, skipping")
            logging.debug(f"[Links.linksFromSites] >>> Profile {u} would produce at least {nlinks} links, skipping")
            skipped.append((u, nlinks))
        else:
            rawlinks = list(itertools.product(*occs))
            for l in rawlinks:
                # l: [(g,c,p,f,u), (g,c,p,f,u), ...)]
                links.append(Link([Occurrence(o[0], o[1], o[2], '+' if o[3] < 3 else '-', o[4]) for o in l], 
                                  int(span), genomes))
            
            #print("[DEBUG] >>> len(l):", len(l))
            #print("[DEBUG] >>>      l:", l)
            #links.extend(l)
            linkProfiles.add((u, nlinks, str(occs)))

    return links, linkProfiles, skipped
