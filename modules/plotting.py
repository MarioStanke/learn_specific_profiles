# functions for plotting and visualization

import cv2
import itertools
import json
import logomaker
import matplotlib.pyplot as plt
import numpy as np
import os

import dataset as ds
import GeneLinkDraw.geneLinkDraw as gld
import model
import sequtils as su



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



def plotLogo(P, idxarray = None, pNames = None, pScores = None, pLosses = None, max_print=5, ax=None):
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
    dfs = su.makeDFs(P.numpy())
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
        
    for p in profileToOcc:
        if (len(profileToOcc[p].keys()) == 1): # or (0 not in profileToOcc[p]):
            continue
            
        occs = []
        for g in profileToOcc[p]:
            occs.append(profileToOcc[p][g])
            
        nlinks = np.prod([len(og) for og in occs])
        if nlinks > linkThreshold:
            print("[DEBUG] >>> Profile", p, "would produce", nlinks, "links, skipping")
            skipped.append((p, nlinks))
        else:
            l = list(itertools.product(*occs))
            #print("[DEBUG] >>> len(l):", len(l))
            #print("[DEBUG] >>>      l:", l)
            links.extend(l)
            linkProfiles.add((p, nlinks, str(occs)))

    return links, linkProfiles, skipped



def drawGeneLinks_simData(genomes, links, posDict, imname, font = "/opt/conda/fonts/Ubuntu-M.ttf", **kwargs):
    """
    Draw an image with simulated "genomes" as horizontal bars and links as connecting lines

    Parameters:
        genomes (list of lists of str): genome sequences
        links (list of lists of occurrences): links between genome positions (occurrences)
        posDict (dict): posDict as returned from genome simulation function
        imname (str): path to image file to write the image to
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
        
    # create links to draw
    drawLinks = []
    for link in links:
        lgenes = []
        lpos = []
        for occ in link:
            gid = str(occ[0])+"_"+str(occ[1])
            lgenes.append(gid)
            lpos.append(occ[2])
            
        drawLinks.append(gld.Link(lgenes, lpos))
        
    img, _ = gld.draw(drawGenes, drawLinks, font = font,
                      genewidth = 20, linkwidth = 1, width = (1920*2), **kwargs)
    img.save(imname)
                        
                        
        
def drawGeneLinks_realData(datapath, seqnames, links, imname, font = "/opt/conda/fonts/Ubuntu-M.ttf", **kwargs):
    """
    Draw an image with genomes as horizontal bars and links as connecting lines from real data

    Parameters:
        datapath (str): path to directory in which the files "orthologs.json"and "hg38.GTF.json" reside
        seqnames (list of lists of str): sequence names, one list per genome
        links (list of lists of occurrences): links between genome positions (occurrences)
        imname (str): path to image file to write the image to
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
        
    img, _ = gld.draw(drawGenes, drawLinks, font = font,
                      genewidth = 20, linkwidth = 1, width = (1920*2),
                      genecols = drawGeneColors, linkcol = palette.color(), **kwargs)
    palette.inc()
    img.save(imname)



def drawGeneLinks_toyData(genomes, links, insertTracking, repeatTracking, imname, 
                          font = "/opt/conda/fonts/Ubuntu-M.ttf", **kwargs):
    """
    Draw an image with toy data "genomes" as horizontal bars and links as connecting lines

    Parameters:
        genomes (list of lists of str): genome sequences
        links (list of lists of occurrences): links between genome positions (occurrences)
        insertTracking (list of lists of dicts): tracking dicts of pattern inserts as returned from toy data creation
        repeatTracking (list of lists of dicts): tracking dicts of repeat inserts as returned from toy data creation
        imname (str): path to image file to write the image to
        font (str): path to the font to use in the image
        **kwargs: named arguments forwarded to gld.draw()
    """
    drawGenes = []
    for g in range(len(genomes)):
        for s in range(len(genomes[g])):
            dgene = gld.Gene(str(g)+"_"+str(s), str(g), len(genomes[g][s]), "+")
            for p in insertTracking[g][s]['pos']:
                dgene.addElement("pattern", p-10, p+10)
                
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
            
    img, _ = gld.draw(drawGenes, drawLinks, font = font,
                      genewidth = 20, linkwidth = 1, width = (1920*2), **kwargs)  
    img.save(imname)



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