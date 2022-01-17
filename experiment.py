import tensorflow as tf
import dataset as dsg
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# similar to model.get_profile_match_sites but applies loss calculations to scores (and max-pools over frames)
def getLossScores(specProModel, pIdx, genomes, ngenomes, tiles_per_X, tile_size, batch_size):
    ds_score = dsg.getDataset(genomes, tiles_per_X, tile_size, True).batch(batch_size).prefetch(3)
    scores = np.ones((ngenomes, max([len(genomes[i][0]) for i in range(ngenomes)])), dtype=float) # score for each position, (N, genomsize)
    
    for X_b, posTrack_b in ds_score:
        for b in range(X_b.shape[0]): # iterate samples in batch
            X = X_b[b]
            posTrack = posTrack_b[b]                                      # (tilesPerX, N, (genomeID, contigID, fwdStart, rcStart))
            
            _, _, Z = specProModel.call(X)                                # (tilesPerX, N, 6, T-k+1, U)
            Z = Z[:,:,:,:,pIdx:(pIdx+1)]                                  # (tilesPerX, N, 6, T-k+1, U) only single profile
            
            tilesPerX = Z.shape[0]
            N = Z.shape[1]
            Tk1 = Z.shape[3]
            
            # apply same normalization as in loss
            Z1 = specProModel._loss_calculation(Z)                        # (N, U, tilesPerX*(T-k+1))
            Z1 = tf.reshape(Z1, [N, -1])                                  # (N, tilesPerX*(T-k+1))

            # restore k-mer score positions in genome
            Idx = tf.broadcast_to(tf.range(0, Tk1, dtype=tf.int32), 
                                  (tilesPerX, N, Tk1))                # (tilesPerX, N, T-k+1) (rel. pos.)
            Idx = tf.transpose(Idx, [1,0,2])                          # (N, tilesPerX, T-k+1)
            Idx = tf.reshape(Idx, [N, -1])                            # (N, tilesPerX*(T-k+1))
            
            
            T = posTrack[:,:,2]                                # (tilesPerX, N) (fwdStart)
            T = tf.repeat(tf.expand_dims(T, -1),
                          Tk1, axis=-1)                        # (tilesPerX, N, T-k+1)
            T = tf.transpose(T, [1,0,2])                       # (N, tilesPerX, T-k+1)
            T = tf.reshape(T, [N, -1])                         # (N, tilesPerX*(T-k+1))
            
            pos = tf.multiply(Idx, 3) # rel. pos * 3           # (N, tilesPerX*(T-k+1))
            pos = tf.add(pos, T) # add tile start
                        
            # store scores in a position matrix
            for n in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    p = pos[n,j]
                    if p >= 0 and p < scores.shape[1]:
                        scores[n,p] = Z1[n,j]

    return scores



# draw an image with genomes as horizontal lines,
#   with markers of inserted patterns (green) and repeats (red)
#   and indicators of loss-scores (top line: pattern profile, bottom: repeat profile)
def drawLossScores(pscores, rscores, genomes, N, insertTracking, repeatTracking, tile_size):
    imgw = int(1920/2)
    imgh = int(1080/2)
    outmarg = 50
    drawh = imgh - (2*outmarg)
    draww = imgw - (2*outmarg)
    im = Image.new('RGBA', (imgw, imgh), 'white')
    draw = ImageDraw.Draw(im)
    
    # draw genomes and tiles
    gendist_y = drawh // (N) # in fact N-1 but reserve space for legend below
    tileh = gendist_y//6
    gen_y = outmarg
    coordDict = {}
    for i in range(N):
        coordDict[i] = {'x1': outmarg, 'x2': draww+outmarg, 'y': gen_y, 'len': draww, 'coord': draww/(len(genomes[i][0])-1)}
        draw.line(((outmarg, gen_y), (outmarg+draww, gen_y)), fill='black', width=5)
        tile_x = outmarg + (coordDict[i]['coord'] * ((tile_size*3)+2))
        j = 1
        while tile_x <= coordDict[i]['x2']:
            draw.line(((tile_x, gen_y-(tileh//2)), (tile_x, gen_y+(tileh//2))), fill='gray', width=1)
            j += 1
            tile_x = outmarg + (coordDict[i]['coord'] * ((j*tile_size*3)+2))
        
        gen_y += gendist_y
        
    # draw repeat and pattern positions
    radius = 8
    for i in insertTracking:
        for pos in insertTracking[i][0]['pos']:
            p = outmarg + (coordDict[i]['coord'] * pos)
            draw.ellipse((p-radius, coordDict[i]['y']-radius, p+radius, coordDict[i]['y']+radius), fill='green')
            
    for i in repeatTracking:
        for pos in repeatTracking[i][0]['pos']:
            p = outmarg + (coordDict[i]['coord'] * pos)
            draw.ellipse((p-radius, coordDict[i]['y']-radius, p+radius, coordDict[i]['y']+radius), fill='red')
            
    # indicate scores for perfect profiles at each position
    pxScores = np.ones((N, max([int(np.floor(coordDict[i]['coord'] * len(genomes[i][0]))) for i in range(N)])), dtype=float) * tf.float32.min # score for each pixel (max binning)
    ppxScores = np.array(pxScores)
    rpxScores = np.array(pxScores)
    for g in range(pxScores.shape[0]):
        for j in range(pscores.shape[1]):
            px = int(np.floor(coordDict[g]['coord'] * j))
            if px < pxScores.shape[1]:
                ppxScores[g,px] = max(ppxScores[g,px], pscores[g,j])
                rpxScores[g,px] = max(rpxScores[g,px], rscores[g,j])
            
    minval = np.min([ppxScores, rpxScores])
    maxval = np.max([ppxScores, rpxScores])
    #minval = specProModel.k * (math.log(specProModel.epsilon))
    #maxval = specProModel.k * (math.log(21))
    if minval != maxval:
        assert minval < maxval, str(minval)+" !< "+str(maxval)
        def gradient(val):
            assert val <= maxval, str(val)+" !<= "+str(maxval)
            p = (val-minval) / (maxval-minval)
            r = int(255 + (p * -255))
            g = int(  0 + (p *  255))
            b = 0
            return (r,g,b)

        for i in range(ppxScores.shape[0]):
            for j in range(ppxScores.shape[1]):
                x = j + outmarg
                py = coordDict[i]['y'] + tileh + 1
                ry = coordDict[i]['y'] + tileh + 4
                draw.point((x, py), fill=gradient(ppxScores[i,j]))
                draw.point((x, ry), fill=gradient(rpxScores[i,j]))
                
        # draw legend
        ly = max([coordDict[i]['y'] for i in coordDict]) + gendist_y
        lx = outmarg
        draw.text((lx, ly+6), str(minval), fill='black')
        vstep = (maxval-minval)/200
        v = minval
        while v <= maxval:
            draw.point((lx, ly), fill=gradient(v))
            lx += 1
            v += vstep
            
        draw.text((lx, ly+6), str(maxval), fill='black')
        
    im = im.resize((imgw*2, imgh*2))
    im.show()
    return im, draw



# some functions to create histograms on huge vectors without draining memory
def ownHist_impl(ls, binSize=None):
    maxls = max(ls)
    minls = min(ls)
    #print("[DEBUG] >>> binSize", binSize, "| maxls", maxls, "| minls", minls, "| (maxls-minls)/100", (maxls-minls)/100)
    if binSize is None:    
        if maxls == minls:
            binSize = 1
        else:
            binSize = max(1, (maxls-minls)/100)
            
    lbin = math.floor(minls/binSize)
    rbin = math.floor(maxls/binSize)
    # intercept if binsize would lead to more than 100 bins
    if len(range(lbin,(rbin+1))) > 101:
        return ownHist_impl(ls, (maxls-minls)/100)
    
    bins = [b*binSize for b in range(lbin,(rbin+1))]
    vals = [0 for b in range(lbin,(rbin+1))]
    for x in ls:
        b = math.floor(x/binSize)
        i = b - lbin
        vals[i] += 1
        
    return bins, vals

def ownHist(ls, binSize=None):
    return ownHist_impl(ls, binSize)

def ownHistRel(ls, binSize=None):
    bins, vals = ownHist(ls, binSize)
    vals[:] = [v/len(ls) for v in vals]
    return bins, vals

def plotOwnHist(bins, vals, ylim=None):
    if ylim is not None:
        assert type(ylim) == tuple, "ylim must be a tuple"
        assert ylim[0] < ylim[1], 'ylim must be a tuple (a, b) where a < b'
        vals = [max(ylim[0], min(ylim[1], y)) for y in vals]
        
    fig, ax = plt.subplots(1, 1, figsize=(16,9))
    ax.bar(x = range(len(bins)),
           height = vals,
           tick_label = bins)
    return fig, ax