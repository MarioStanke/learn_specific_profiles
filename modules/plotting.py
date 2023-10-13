# functions for plotting and visualization

import cv2
import itertools
import json
import logging
import logomaker
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

import GeneLinkDraw.geneLinkDraw as gld
import Links
import SequenceRepresentation
import sequtils as su

# set logging level for logomaker to avoid debug message clutter
logging.getLogger('logomaker').setLevel(logging.WARNING)


def plotHistory(history):
    """ 
    Plot the training history as loss and accuracy curves 
    
    Parameters
        history (dict): History dict from trained model

    Returns
        fig, ax: matplotlib figure and axes objects
    """

    loss = history['loss']
    Rmax = history['Rmax']
    Rmin = history['Rmin']
    Smax = history['Smax']
    Smin = history['Smin']
    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(ncols = 2, figsize = (15, 6))
    ax[0].plot(epochs, loss, 'bo', label = 'Training loss')
    ax[0].set_title('Training loss')
    ax[0].legend()

    ax[1].plot(epochs, Rmax, 'bo', label = 'Rmax')
    ax[1].plot(epochs, Rmin, 'b+', label = 'Rmin')
    ax[1].plot(epochs, Smax, 'go', label = 'Smax')
    ax[1].plot(epochs, Smin, 'g+', label = 'Smin')
    ax[1].set_title('Training R and S')
    ax[1].legend()
    
    return fig, ax



def plotLogo(P: np.ndarray, idxarray = None, pNames = None, pScores = None, pLosses = None, max_print=5, ax=None):
    """
    Create logo(s) from profile(s)

    Parameters
        P (tf.Tensor): tensor of shape (k, alphabet_size, U)
        idxarray (list of int): optional list of indices in P's third axis to plot only certain profiles
        pNames (list): optional list of names to assign for each profile in P (if None, index of profile will be 
                       displayed)
        pScores (list of float): optional list of scores for each profile in P
        pLosses (list of float): optional list of losses for each profile in P
        max_print (int): print up to this many logos
        ax (list of matplotlib axes): optional axes from a matplotlib Figure to plot on, one axes for each plot
    """
    if type(P) is not np.ndarray: # catch tf.Tensor
        P = P.numpy()

    dfs = su.makeDFs(P)
    for i in range(min(P.shape[2], max_print)):
        j = idxarray[i] if idxarray is not None else i
        profile_df = dfs[j]
        if ax is None:
            logo = logomaker.Logo(profile_df, vpad=.1, width=1)
        else:
            logo = logomaker.Logo(profile_df, vpad=.1, width=1, ax=ax[i])
            
        logo.style_xticks(anchor=0, spacing=1, rotation=45)
        logo.ax.set_ylabel('information (bits)')
        nametext = "Profile "+str(pNames[j]) if pNames is not None else f"Profile {j}"
        scoretext = (f" score={pScores[j]:.3f}") if pScores is not None else ""
        losstext = (f" loss={pLosses[j]:.3f}") if pLosses is not None else ""
        logo.ax.set_title(nametext + scoretext + losstext)



def sitesToLinks(sites, linkThreshold = 100):
    """
    From a tensor of profile match sites, create a list of links to use in drawGeneLinks_*()

    Parameters
        sites (np.ndarray): array of shape (sites, (genomeID, contigID, pos, u, f)) with the profile match sites
        linkThreshold (int): do not create links from a profile that would result in more than this many links

    Returns
        links: list of links
        linkProfiles: set of tuples with information from what the links were created
        skipped: list of profiles that have been skipped due to too many links
    """
    # sites.shape == (fwdSites, (genomeID, contigID, pos, u, f))
    links = []
    skipped = []
    profileToOcc = {}
    linkProfiles = set()
    for g, c, p, u, _ in sites:
        if u not in profileToOcc:
            profileToOcc[u] = {}
            
        if g not in profileToOcc[u]:
            profileToOcc[u][g] = []
            
        profileToOcc[u][g].append([g,c,p])
        
    for u in profileToOcc:
        if (len(profileToOcc[u].keys()) == 1): # or (0 not in profileToOcc[p]):
            continue
            
        occs = []
        for g in profileToOcc[u]:
            occs.append(profileToOcc[u][g])
            
        # nlinks = np.prod([len(og) for og in occs]) # does not handle overflow!
        nlinks = 1
        for og in occs:
            nlinks *= len(og)
            if nlinks > linkThreshold:
                break

        if nlinks > linkThreshold:
            #print("[DEBUG] >>> Profile", u, "would produce at least", nlinks, "links, skipping")
            logging.warning(f"[plotting.sitesToLings] >>> Profile {u} would produce at least {nlinks} links, skipping")
            skipped.append((u, nlinks))
        else:
            l = list(itertools.product(*occs))
            #print("[DEBUG] >>> len(l):", len(l))
            #print("[DEBUG] >>>      l:", l)
            links.extend(l)
            linkProfiles.add((u, nlinks, str(occs)))

    return links, linkProfiles, skipped



def drawGeneLinks_SequenceRepresentationData(genomes: list[SequenceRepresentation.Genome], 
                                             links: list[Links.Link], imname,
                                             kmerSites = None, kmerCol = None, 
                                             maskingSites = None, maskingCol = 'darkred',
                                             font = "/opt/conda/fonts/Ubuntu-M.ttf", **kwargs):
    """
    Draw an image with sequences as horizontal bars and links as connecting lines

    Parameters:
        genomes (list of SequenceRepresentation.Genome): genomes to draw
        links (list of Links.Link): links between sequence positions (occurrences)
        imname (str): path to image file to write the image to, set to None for no writing
        kmerSites (list of tuples): Optional list of tuples of format (genomeID, contigID, pos) of initial kmer 
                                    positions. If given, these occurrences are drawn as small dots on the genes
        kmerCol (str or tuple of RGB values): Optional color used when drawing initial kmer positions. If None and
                                              kmerSites are given, color is determined automatically. Use gld.Palette
                                              class to get some color names
        maskingSites (list of tuples): Optional list of tuples of format (genomeID, contigID, pos) of sites where DNA
                                       was softmasked during training
        maskingCol (str or tuple of RGB values): Optional color used when drawing masking sites. Defaults to darkred. If
                                                 None and maskingSites are given, color is determined automatically. Use
                                                 gld.Palette class to get some color names
        font (str): path to font file to use for text
        **kwargs: named arguments forwarded to gld.draw()
    """

    drawGenes = []
    for genome in genomes:
        for sequence in genome:
            dg = gld.Gene(sequence.id, sequence.species, sequence.length, sequence.strand)
            for element in sequence.genomic_elements:
                start, end = element.getRelativePositions(sequence, from_rc = False) # Always drawing forward strand
                dg.addElement(element.type, start, end-1) # TODO: refactor gld to also use exclusive end positions!

            drawGenes.append(dg)

    # for some assertions
    geneids = []
    for dg in drawGenes:
        geneids.append(dg.id)
    geneids = sorted(geneids)
    assert len(geneids) == len(set(geneids)), "[ERROR] >>> Duplicate gene ids in "+str(geneids)
        
    # create links to draw
    drawLinks = []
    for link in links:
        lgenes = []
        lpos = []
        for occ in link:
            gid = genomes[int(occ.genomeIdx)][int(occ.sequenceIdx)].id
            assert gid in geneids, f"[ERROR] >>> gene id {gid} from occurrence {occ} not found in {geneids}"
            lgenes.append(gid)
            lpos.append(occ.position)
            
        drawLinks.append(gld.Link(lgenes, lpos))

    # also create kmer-"Link" showing the position of initial kmers and/or masking-"Link" to see where masking happened
    def createAdditionalSites(sites, col):
        occDict = {}
        for site in sites:
            gid = genomes[int(site[0])][int(site[1])].id
            assert gid in geneids, f"[ERROR] >>> gene id {gid} from occurrence {site} not found in {geneids}"
            if gid not in occDict:
                occDict[gid] = []

            occDict[gid].append(site[2])

        if len(occDict.keys()) >= 2:
            # no links possible if less than two genomes
            lgenes = []
            lpos = []
            for gid in geneids:
                if gid in occDict:
                    lgenes.append(gid)
                    lpos.append(occDict[gid])

            drawLinks.append(gld.Link(lgenes, lpos, connect=False, compressed=True, color=col))
        else:
            #print("[WARNING] >>> Could not create kmer sites or masking sites because less than 2 genes are involved")
            logging.warning("[plotting.drawGeneLinks_SequenceRepresentationData.createAdditionalSites] >>> Could not "+\
                            "create kmer sites or masking sites because less than 2 genes are involved")

    if kmerSites is not None:
        createAdditionalSites(kmerSites, kmerCol)
    if maskingSites is not None:
        createAdditionalSites(maskingSites, maskingCol)


    # avoid masking kwargs and set defaults here that can be overwritten in function call
    gw = 20 if 'genewidth' not in kwargs else kwargs['genewidth']
    kwargs.pop('genewidth') if 'genewidth' in kwargs else ()
    lw = 1  if 'linkwidth' not in kwargs else kwargs['linkwidth']
    kwargs.pop('linkwidth') if 'linkwidth' in kwargs else ()
    img, _ = gld.draw(drawGenes, drawLinks, font = font,
                      genewidth = gw, linkwidth = lw, #width = (1920*2), 
                      **kwargs)
    if imname:
        img.save(imname)

    img.close()




def drawGeneLinks_simData(genomes, links, posDict, imname, linksAreSites = False, 
                          kmerSites = None, kmerCol = None, maskingSites = None, maskingCol = 'darkred',
                          highlightProfiles = None, highlightcol = 'darkgreen',
                          font = "/opt/conda/fonts/Ubuntu-M.ttf", **kwargs):
    """
    Draw an image with simulated "genomes" as horizontal bars and links as connecting lines

    Parameters:
        genomes (list of lists of str): genome sequences
        links (list of lists of occurrences): links between genome positions (occurrences). Can alternatively be the
                                              sites (np.ndarray of shape (sites, (genomeID, contigID, pos, u, f)) with 
                                              the profile match sites), then linksAreSites must be True
        posDict (dict): posDict as returned from genome simulation function
        imname (str): path to image file to write the image to, set to None for no writing
        linksAreSites (bool): Must be set to True if links are acutally sites, otherwise must be False (default)
        kmerSites (list of tuples): Optional list of tuples of format (genomeID, contigID, pos) of initial kmer 
                                    positions. If given, these occurrences are drawn as small dots on the genes
        kmerCol (str or tuple of RGB values): Optional color used when drawing initial kmer positions. If None and
                                              kmerSites are given, color is determined automatically. Use gld.Palette
                                              class to get some color names
        maskingSites (list of tuples): Optional list of tuples of format (genomeID, contigID, pos) of sites where DNA
                                       was softmasked during training
        maskingCol (str or tuple of RGB values): Optional color used when drawing masking sites. Defaults to darkred. If
                                                 None and maskingSites are given, color is determined automatically. Use
                                                 gld.Palette class to get some color names
        highlightProfiles (list of int): Optional list of profile indices, only works if linksAreSites is True! Use it
                                         to define profile indices (u) in the site tuples that get a different link 
                                         color
        highlightcol (str or tuple of RGB values): Color to use for highlighted links
        font (str): path to the font to use in the image
        **kwargs: named arguments forwarded to gld.draw()
    """
    drawGenes = []
    for g in range(len(genomes)):
        for c in range(len(genomes[g])):
            dg = gld.Gene(str(g)+"_"+str(c), str(g), len(genomes[g][c]), "+")
            drawGenes.append(dg)
                    
    for dgene in drawGenes:
        dgene.addElement("gene", posDict['start_codon'], posDict['stop_codon']+2)

    # for some assertions
    geneids = []
    for dg in drawGenes:
        geneids.append(dg.id)
    geneids = sorted(geneids)
    assert len(geneids) == len(set(geneids)), "[ERROR] >>> Duplicate gene ids in "+str(geneids)
        
    # create links to draw
    drawLinks = []
    if not linksAreSites:
        for link in links:
            lgenes = []
            lpos = []
            for occ in link:
                gid = str(occ[0])+"_"+str(occ[1])
                assert gid in geneids, "[ERROR] >>> gene id {} from occurrence {} not found in {}".format(gid, str(occ),
                                                                                                          str(geneids))
                lgenes.append(gid)
                lpos.append(occ[2])
                
            drawLinks.append(gld.Link(lgenes, lpos))
    else:
        # links: array of shape (sites, (genomeID, contigID, pos, u, f)) with the profile match sites
        clinkDict = {}
        for g, c, p, u, _ in links:
            if u not in clinkDict:
                clinkDict[u] = {}

            gid = str(g)+"_"+str(c)
            assert gid in geneids, "[ERROR] >>> gene id {} from site {} not found in {}".format(gid, str((g,c,p,u)),
                                                                                                str(geneids))
            if gid not in clinkDict[u]:
                clinkDict[u][gid] = []

            clinkDict[u][gid].append(p)

        for u in clinkDict:
            if len(clinkDict[u].keys()) < 2:
                continue # no links possible if less than two genomes

            lgenes = []
            lpos = []
            # create sorted, compressed Links
            for gid in geneids:
                if gid in clinkDict[u]:
                    lgenes.append(gid)
                    lpos.append(clinkDict[u][gid])

            linkcol = highlightcol if highlightProfiles is not None and u in highlightProfiles else None
            drawLinks.append(gld.Link(lgenes, lpos, compressed=True, color=linkcol))

    # also create kmer-"Link" showing the position of initial kmers and/or masking-"Link" to see where masking happened
    def createAdditionalSites(sites, col):
        occDict = {}
        for site in sites:
            gid = str(site[0])+"_"+str(site[1])
            assert gid in geneids, "[ERROR] >>> gene id {} from kmer site {} not found in {}".format(gid, 
                                                                                           str((g,c,p,u)), str(geneids))
            if gid not in occDict:
                occDict[gid] = []

            occDict[gid].append(site[2])

        if len(occDict.keys()) >= 2:
            # no links possible if less than two genomes
            lgenes = []
            lpos = []
            for gid in geneids:
                if gid in occDict:
                    lgenes.append(gid)
                    lpos.append(occDict[gid])

            drawLinks.append(gld.Link(lgenes, lpos, connect=False, compressed=True, color=col))
        else:
            #print("[WARNING] >>> Could not create kmer sites or masking sites because less than 2 genes are involved")
            logging.warning("[plotting.drawGeneLinks_simData.createAdditionalSites] >>> Could not create kmer sites " +\
                            "or masking sites because less than 2 genes are involved")

    if kmerSites is not None:
        createAdditionalSites(kmerSites, kmerCol)
    if maskingSites is not None:
        createAdditionalSites(maskingSites, maskingCol)


    # avoid masking kwargs and set defaults here that can be overwritten in function call
    gw = 20 if 'genewidth' not in kwargs else kwargs['genewidth']
    kwargs.pop('genewidth') if 'genewidth' in kwargs else ()
    lw = 1  if 'linkwidth' not in kwargs else kwargs['linkwidth']
    kwargs.pop('linkwidth') if 'linkwidth' in kwargs else ()
    img, _ = gld.draw(drawGenes, drawLinks, font = font,
                      genewidth = gw, linkwidth = lw, #width = (1920*2), 
                      **kwargs)
    if imname:
        img.save(imname)

    img.close()
                        
                        
        
def drawGeneLinks_realData(datapath, seqnames, links, imname, font = "/opt/conda/fonts/Ubuntu-M.ttf", **kwargs):
    """
    Draw an image with genomes as horizontal bars and links as connecting lines from real data

    Parameters:
        datapath (str): path to directory in which the files "orthologs.json"and "hg38.GTF.json" reside
        seqnames (list of lists of str): sequence names, one list per genome
        links (list of lists of occurrences): links between genome positions (occurrences)
        imname (str): path to image file to write the image to, set to None for no writing
        font (str): path to the font to use in the image
        **kwargs: named arguments forwarded to gld.draw()
    """
    with open(os.path.join(datapath, "orthologs.json"), 'rt') as fh:
        orthology = json.load(fh)
        
    with open(os.path.join(datapath, "hg38.GTF.json"), 'rt') as fh:
        gtf = json.load(fh)
        
    def parseSequenceHeader(header):
        fields = header.split("|")
        headDict = {}
        for field in fields:
            key, value = field.split(":")
            assert len(field.split(":")) == 2, "[ERROR] >>> could not parse header: "+header
            assert key not in headDict, "[ERROR] >>> could not parse header: "+header
            headDict[key] = value

        return headDict

    geneColors = {}
    palette = gld.Palette()
    # first, color all orthologs the same
    for i in range(len(seqnames)):
        for sid in seqnames[i]:
            if sid in orthology and sid not in geneColors:
                geneColors[sid] = palette.color()
                for seq in orthology[sid]:
                    assert seq not in geneColors, str(seq)
                    geneColors[seq] = palette.color()
                    
                palette.inc()
                
    # color all artificials the same
    for i in range(len(seqnames)):
        for sid in seqnames[i]:
            if sid not in geneColors and parseSequenceHeader(sid)['gid'] == 'artificial':
                geneColors[sid] = palette.color()
                
    palette.inc()
    
    # color any remaining sequences
    for i in range(len(seqnames)):
        for sid in seqnames[i]:
            if sid not in geneColors:
                geneColors[sid] = palette.color()
    
    palette.inc()
    
    # create gene and color lists for later drawing
    # first, sort seqnames: first the real sequences, then the artificials, 
    #   such that ortholog sequences have the same index in all genomes
    genomeOrder = []
    for seqs in seqnames:
        genomeOrder.append(parseSequenceHeader(seqs[0])['genome'])
        
    sortedSeqnames = [[] for _ in seqnames]
    for sid in seqnames[0]: # based on reference (i.e. first genome)
        head = parseSequenceHeader(sid)
        if head['gid'] != 'artificial':
            sortedSeqnames[0].append(sid)
            for o in orthology[sid]:
                ohead = parseSequenceHeader(o)
                idx = genomeOrder.index(ohead['genome'])
                sortedSeqnames[idx].append(o)
                
    for i in range(len(seqnames)):
        for sid in seqnames[i]:
            head = parseSequenceHeader(sid)
            if head['gid'] == 'artificial':
                sortedSeqnames[i].append(sid)
                
    # assert that all sequence names are kept
    for i in range(len(seqnames)):
        assert sorted(seqnames[i]) == sorted(sortedSeqnames[i]), "Something went wrong in genome "+str(i)
            
    
    drawGenes = []
    drawGeneColors = []
    for i in range(len(sortedSeqnames)):
        for sid in sortedSeqnames[i]:
            head = parseSequenceHeader(sid)
            dg = gld.Gene(head['genome']+"_"+head['tid'], head['genome'], int(head['seqlen']), head['strand'])
            dg._tid = head['tid'] # add a special field containing just the tid for element addition
            drawGenes.append(dg)
            drawGeneColors.append(geneColors[sid])
            
    # assert no duplicate geneIDs
    geneIDs = [dg.id for dg in drawGenes]
    assert len(geneIDs) == len(set(geneIDs)), "geneIDs contain duplicates"
            
    # add CDS etc.
    for dgene in drawGenes:
        if getattr(dgene, '_tid', None) in gtf:
            elems = gtf[dgene._tid]
            for elem in elems:
                if elem['feature'] == "CDS":
                    dgene.addElement(elem['feature'], elem['rstart'], elem['rend'])

    # create links to draw
    tidToDrawgenes = {}
    for dgene in drawGenes:
        tidToDrawgenes[dgene.id] = dgene
        
    drawLinks = []
    for link in links:
        lgenes = []
        lpos = []
        for occ in link:
            sid = seqnames[occ[0]][occ[1]]
            head = parseSequenceHeader(sid)
            lgenes.append(tidToDrawgenes[head['genome']+"_"+head['tid']].id)
            lpos.append(occ[2])
            
        drawLinks.append(gld.Link(lgenes, lpos))
        
    # avoid masking kwargs and set defaults here that can be overwritten in function call
    gw = 20 if 'genewidth' not in kwargs else kwargs['genewidth']
    kwargs.pop('genewidth') if 'genewidth' in kwargs else ()
    lw = 1  if 'linkwidth' not in kwargs else kwargs['linkwidth']
    kwargs.pop('linkwidth') if 'linkwidth' in kwargs else ()
    lc = palette.color() if 'linkcol' not in kwargs else kwargs['linkcol']
    kwargs.pop('linkcol') if 'linkcol' in kwargs else ()
    img, _ = gld.draw(drawGenes, drawLinks, font = font,
                      genewidth = gw, linkwidth = lw, #width = (1920*2),
                      genecols = drawGeneColors, linkcol = lc, **kwargs)
    palette.inc()
    if imname:
        img.save(imname)

    img.close()



def drawGeneLinks_toyData(genomes, links, insertTracking, repeatTracking, imname, 
                          font = "/opt/conda/fonts/Ubuntu-M.ttf", **kwargs):
    """
    Draw an image with toy data "genomes" as horizontal bars and links as connecting lines

    Parameters:
        genomes (list of lists of str): genome sequences
        links (list of lists of occurrences): links between genome positions (occurrences)
        insertTracking (list of lists of dicts): tracking dicts of pattern inserts as returned from toy data creation
        repeatTracking (list of lists of dicts): tracking dicts of repeat inserts as returned from toy data creation
        imname (str): path to image file to write the image to, set to None for no writing
        font (str): path to the font to use in the image
        **kwargs: named arguments forwarded to gld.draw()
    """
    drawGenes = []
    for g in range(len(genomes)):
        for s in range(len(genomes[g])):
            dgene = gld.Gene(str(g)+"_"+str(s), str(g), len(genomes[g][s]), "+")
            if len(insertTracking) > 0:
                for p in insertTracking[g][s]['pos']:
                    dgene.addElement("pattern", p-10, p+10)

            if len(repeatTracking) > 0:
                for p in repeatTracking[g][s]['pos']:
                    dgene.addElement("repeat", p-10, p+10)
            
            drawGenes.append(dgene)
            
    drawLinks = []
    for link in links:
        lgenes = []
        lpos = []
        for occ in link:
            gid = str(occ[0])+"_"+str(occ[1])
            lgenes.append(gid)
            lpos.append(occ[2])
            
        drawLinks.append(gld.Link(lgenes, lpos))
            
    # avoid masking kwargs and set defaults here that can be overwritten in function call
    gw = 20 if 'genewidth' not in kwargs else kwargs['genewidth']
    kwargs.pop('genewidth') if 'genewidth' in kwargs else ()
    lw = 1  if 'linkwidth' not in kwargs else kwargs['linkwidth']
    kwargs.pop('linkwidth') if 'linkwidth' in kwargs else ()
    img, _ = gld.draw(drawGenes, drawLinks, font = font,
                      genewidth = gw, linkwidth = lw, #width = (1920*2), 
                      **kwargs)  
    if imname:
        img.save(imname)

    img.close()



def makeVideo(path, video_name, fps = 1):
    """
    Given a directory containing images, create a video from these images and store it there

    Parameters
        path (str): path to the directory containing the images, video is stored here
        video_name (str): filename of the created video
        fps (int): frames per second, i.e. how many images are shown in the video per second
    """

    assert os.path.isdir(path), str(path)+" not found or is not a directory"

    images = sorted([img for img in os.listdir(path) if img.endswith("png")])
    frame = cv2.imread(os.path.join(path, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, _ = frame.shape

    video = cv2.VideoWriter(os.path.join(path, video_name), cv2.VideoWriter_fourcc(*'H264'), fps, (width, height))

    # Appending the images to the video one by one
    for image in images:
        video.write(cv2.imread(os.path.join(path, image)))

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()



def combinePlots(plots: list, 
                 rows: int, cols: int, 
                 out: str, 
                 space=0, labels=None, fontsize=36, fontpath=None,
                 **kwargs):
    """ 
    General function to concatenate plots (from disk) to a multi-plot image

    Parameters
        plots (list of str): paths to the images that should be concatenated
        rows (int): number of vertical plots
        cols (int): number of horizontal plots
        out (str): path to the output image
        space (int): space between two plots in pixels
        labels (list of str): optional list of plot labels (e.g. ['A)', 'B)'])
        fontsize (int): fontsize in pt
        fontpath (str): optional path to the desired font
        **kwargs: keyword arguments passed to PIL.Image.save() 
    """

    assert len(plots) <= rows*cols, "Too many plots"
    assert len(plots) > (rows-1)*cols, "Too few plots, reduce rows or columns"
    if labels is not None:
        if fontpath is not None:
            assert os.path.isfile(fontpath), "Font not found"
        else:
            fontpath = '/usr/share/fonts/truetype/freefont/FreeSerif.ttf'

        if not os.path.isfile(fontpath):
            addLabels = False
            #print("[WARNING] >>> Ignoring labels, default fontpath not found: "+fontpath)
            logging.warning(f"[plotting.combinePlots] >>> Ignoring labels, default fontpath not found: {fontpath}")
        else:
            addLabels = True
            font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', fontsize)
    else:
        addLabels = False
        
    if addLabels:
        assert len(labels) == len(plots), "provide one label for each plot"
    
    # load images
    images = [Image.open(p) for p in plots]
    
    # arrange images row-wise
    arrange = np.empty([rows, cols, 3]) # last: [idx, w, h]
    arrange *= np.nan
    k = 0
    for i in range(rows):
        for j in range(cols):
            if k < len(plots):
                arrange[i,j,0] = k
                arrange[i,j,1] = images[k].width
                arrange[i,j,2] = images[k].height
                k += 1
        
    # get canvas dimensions    
    canvasWidth = int(max(np.nansum(arrange[:,:,1], axis=1)) + space*(cols-1))
    canvasHeight = int(max(np.nansum(arrange[:,:,2], axis=0)) + space*(rows-1))
    
    # assemble images
    im = Image.new('RGB', (canvasWidth, canvasHeight), color='white')
    if addLabels:
        draw = ImageDraw.Draw(im)
        
    y = 0
    for i in range(rows):
        x = 0
        y += 0 if i == 0 else int(np.nanmax(arrange[(i-1),:,2])) + space # add row heights
        for j in range(cols):
            x += 0 if j == 0 else int(np.nanmax(arrange[:,(j-1),1])) + space # add col widhts
            imgIdx = arrange[i,j,0]
            if not np.isnan(imgIdx):
                im.paste(images[int(imgIdx)], (x,y))
            if addLabels:
                draw.text((x,y), labels[int(imgIdx)], 'black', font)
    
    im.save(out, **kwargs)
    im.close()