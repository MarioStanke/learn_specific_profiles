# All sorts of helper functions that don't fit elsewhere

import itertools
import logging
import numpy as np

import plotting
import sequtils as su

# ---------------------
# kmer/profile tracking
# ---------------------

class kmerOccurrence:
    def __init__(self, g, c, pos):
        self.genome = g
        self.contig = c
        self.pos = pos



def kmerMidpoint(i,k):
    """ given a start position i and a k, return the midpoint of the kmer """
    m = i + (k//2)
    return m



def sitesToLinks(sites, linkThreshold = 100, kmer=""):
    """
    Create links from a list of occurrences

    Parameters
        sites (list of list of kmerOccurrence): sites to create links from
        linkThreshold (int): do not create links if more than this number of links would be created
        kmer (str): optional kmer string for debug message

    Returns
        links: list of links
    """
    # sites == [[occ0.0, occ0.1, ...], [occ1.0, ...], ...]
    links = []
    nlinks = np.prod([len(og) for og in sites])
    if nlinks > linkThreshold:
        prodstr = ' * '.join([str(len(og)) for og in sites])
        #print("[DEBUG] >>> "+kmer+" would produce", nlinks, "links ("+prodstr+"), skipping")
        logging.debug(f"[stuff.sitesToLinks] >>> {kmer} would produce {nlinks} links ({prodstr}), skipping")
        return None
        
    l = list(itertools.product(*sites))
    links.extend(l)

    return links



def findGeneSpanningKmer(N, occs, posDict, returnLink = False, kmer=""):
    """
    Given a list of kmerOccurences, find the kmers that span all genes in the simulated genomes

    Parameters
        N (int): total number of simulated genomes
        occs (list of kmerOccurrence): list of kmer occurrences
        posDict (dict): posDict as returned by genome simulation function
        returnLink (bool): if True, returns the links from the occurrences, otherwise (default) returns True if the
                           occs span all genes, False otherwise
        kmer (str): optional kmer string for debug message
    """
    # separate occs after genome
    genomeOccs = [[] for _ in range(N)]
    for occ in occs:
        assert occ.contig == 0, "[ERROR] >>> Currently not able to run on genomes with > 1 sequence/contig"
        genomeOccs[occ.genome].append(occ)
    
    # if not all genomes covered, we're done
    if not all([len(o) for o in genomeOccs]):
        return False if not returnLink else None
    
    # check if there is an occurrence inside the CDS in all genomes
    a = posDict['cds_start']
    b = a + posDict['cds_len'] - 1
    linkOccs = None if not returnLink else [[] for _ in range(N)]
    for go in genomeOccs:
        # as soon as one genome is found that has no occurrence inside CDS, we're done
        if not any([occ.pos >= a and occ.pos <= b for occ in go]):
            return False if not returnLink else None
        
        if returnLink:
            o = np.array(go)
            i = [occ.pos >= a and occ.pos <= b for occ in o]
            linkOccs[o[0].genome].extend(list(o[i])) # store occurrences inside CDS
        
    # at least one occurrence per genome that is inside CDS
    if not returnLink:
        return True
    
    else:
        # create links inside CDS
        return sitesToLinks(linkOccs, kmer=kmer)



def checkKmers(genomes, posDict, kmers, draw=False, history=None, verbose=True):
    """ 
    Perform the check of kmers if they span all genes

    Parameters:
        genomes (list of lists of str): genomes
        posDict (dict): posDict as returned by genome simulation function
        kmers (list of str): kmers to check
        draw: boolean, if True, create a genome-link-image with the found links inside CDS (if any), might not work
        history: genome list from previous run, to optionally check if new generated sequences are different 

    Returns:
        contained: list of kmer indices that span all genes
    """
    
    if len(kmers) == 0:
        #print("[WARNING] >>> no kmers given")
        logging.warning("[stuff.checkKmers] >>> no kmers given")
        return None, None

    # store kmer occurrences
    assert all([len(kmer) == len(kmers[0]) for kmer in kmers]), "[ERROR] >>> not all kmers have same k"
    k = len(kmers[0])
    kmerOccs = {}
    for kmer in kmers:
        kmerOccs[kmer] = []

    for g in range(len(genomes)):
        for c in range(len(genomes[g])):
            sft = su.six_frame_translation(genomes[g][c])
            for f in range(len(sft)):
                seq = sft[f]
                for i in range(len(seq)-k+1):
                    kmer = seq[i:i+k]
                    if kmer in kmers: # only check relevant kmers
                        aapos = kmerMidpoint(i,k)
                        dnapos = su.convert_six_frame_position(aapos, f, len(genomes[g][c]), dna_to_aa=False)
                        kmerOccs[kmer].append(kmerOccurrence(g,c,dnapos))
                            
    # check if selected kmers contain an all-gene-spanning kmer
    contained = []
    links = []
    for i in range(len(kmers)):
        kmer = kmers[i]
        found = findGeneSpanningKmer(len(genomes), kmerOccs[kmer], posDict, returnLink=draw, kmer=kmer)
        if found:
            contained.append(i)
            
        if draw and found is not None:
            links.extend(found)
            
    if draw:
        plotting.drawGeneLinks_simData(genomes, links, posDict, None)
        
    return contained