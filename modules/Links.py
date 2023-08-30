from dataclasses import dataclass
import itertools
import numpy as np

#import model
import SequenceRepresentation
import sequtils as su

@dataclass
class Occurrence:
    genomeIdx: int
    sequenceIdx: int
    position: int
    frame: int
    profileIdx: int

    def __str__(self) -> str:
        return(f"(g: {self.genomeIdx}, c: {self.sequenceIdx}, p: {self.position}, f: {self.frame}, "\
               + f"u: {self.profileIdx})")
    def __repr__(self) -> str:
        return self.__str__()
        #return (f"Occurrence(genomeID={self.genomeIdx}, sequenceID={self.sequenceIdx}, position={self.position}, " \
        #        +f"frame={self.frame}, profileID={self.profileIdx})")
    def __lt__(self, other):
        return self.tuple() < other.tuple()
    def __eq__(self, other):
        return self.tuple() == other.tuple()
    def tuple(self):
        return (self.genomeIdx, self.sequenceIdx, self.position, self.frame, self.profileIdx)


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

    def __post_init__(self):
        assert self.span > 0, "[ERROR] >>> Span must be positive: " + str(self.span)
        self.occs = sorted(self.occs)
        genoccs = [o.genomeIdx for o in self.occs]
        assert genoccs == sorted(set(genoccs)), "[ERROR] >>> Genomes are not unique: " + str(genoccs)
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
            if occ.frame >= 3:
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
    for g, c, p, u, f in sites:
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
            print("[DEBUG] >>> Profile", u, "would produce at least", nlinks, "links, skipping")
            skipped.append((u, nlinks))
        else:
            rawlinks = list(itertools.product(*occs))
            for l in rawlinks:
                # l: [(g,c,p,f,u), (g,c,p,f,u), ...)]
                links.append(Link([Occurrence(o[0], o[1], o[2], o[3], o[4]) for o in l], span, genomes))
            
            #print("[DEBUG] >>> len(l):", len(l))
            #print("[DEBUG] >>>      l:", l)
            #links.extend(l)
            linkProfiles.add((u, nlinks, str(occs)))

    return links, linkProfiles, skipped
