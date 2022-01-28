import tensorflow as tf
import dataset as dsg
import numpy as np
import math
import matplotlib.pyplot as plt
import model
from PIL import Image, ImageDraw

# only get list of all loss scores without position information
#def getLossScores_raw(specProModel, pIdx, genomes, ngenomes, tiles_per_X, tile_size, batch_size):
#    ds_score = dsg.getDataset(genomes, tiles_per_X, tile_size, True).batch(batch_size).prefetch(3)
#    scores = np.empty((ngenomes, max([len(genomes[i][0]) for i in range(ngenomes)])), dtype=float) # score for each position, (N, genomsize)
#    scores[:,:] = np.NaN
#    for X_b, posTrack_b in ds_score:
#        for b in range(X_b.shape[0]): # iterate samples in batch
#            X = X_b[b]                                                    # (tilesPerX, N, 6, T, 21)
#            posTrack = posTrack_b[b]                                      # (tilesPerX, N, (genomeID, contigID, fwdStart, rcStart))
#            
#            _, _, Z = specProModel.call(X)                                # (tilesPerX, N, 6, T-k+1, U)
#            Z = Z[:,:,:,:,pIdx:(pIdx+1)]                                  # (tilesPerX, N, 6, T-k+1, U) only single profile
#            #print("[DEBUG] >>> Z.shape:", Z.shape)
#            tilesPerX = Z.shape[0]
#            N = Z.shape[1]
#            Tk1 = Z.shape[3]
#            k = specProModel.k
#            
#            # apply same normalization as in loss
#            Z1 = specProModel._loss_calculation(Z)                        # (N, U, tilesPerX*6*(T-k+1))


# similar to model.get_profile_match_sites but applies loss calculations to scores (and max-pools over frames)
def getLossScores(specProModel, pIdx, genomes, ngenomes, tiles_per_X, tile_size, batch_size):
    ds_score = dsg.getDataset(genomes, tiles_per_X, tile_size, True).batch(batch_size).prefetch(3)
    #scores = np.ones((ngenomes, max([len(genomes[i][0]) for i in range(ngenomes)])), dtype=float) # score for each position, (N, genomsize)
    scores = np.empty((ngenomes, max([len(genomes[i][0]) for i in range(ngenomes)])), dtype=float) # score for each position, (N, genomsize)
    scores[:,:] = np.NaN
    #print("[DEBUG] >>> scores:", scores)
    for X_b, posTrack_b in ds_score:
        for b in range(X_b.shape[0]): # iterate samples in batch
            X = X_b[b]
            posTrack = posTrack_b[b]                                      # (tilesPerX, N, (genomeID, contigID, fwdStart, rcStart))
            #print("[DEBUG] >>> X.shape:", X.shape)
            #print("[DEBUG] >>> posTrack:", posTrack)
            
            _, _, Z = specProModel.call(X)                                # (tilesPerX, N, 6, T-k+1, U)
            Z = Z[:,:,:,:,pIdx:(pIdx+1)]                                  # (tilesPerX, N, 6, T-k+1, U) only single profile
            #print("[DEBUG] >>> Z.shape:", Z.shape)
            tilesPerX = Z.shape[0]
            N = Z.shape[1]
            Tk1 = Z.shape[3]
            k = specProModel.k
            
            # apply same normalization as in loss
            Z1 = specProModel._loss_calculation(Z)                        # (N, U, tilesPerX*6*(T-k+1))
            #print("[DEBUG] >>> Z1:", Z1)
            Z1 = tf.reshape(Z1, [N, -1])                                  # (N, tilesPerX*6*(T-k+1))
            #print("[DEBUG] >>> Z1 reshape:", Z1)

            # restore k-mer score positions in genome
            Idx = model.indexTensor(tilesPerX, N, 6, Tk1, 1)    # shape (tilesPerX, N, 6, T-k+1, U, (f,p,u) (frame, rel.pos., profile)
            Idx = tf.reshape(Idx, [tilesPerX, N, 6, Tk1, -1])   # shape (tilesPerX, N, 6, T-k+1, (f,p,u))
            #print("[DEBUG] >>> Idx:", Idx)
            Idx = tf.transpose(Idx, [1,0,2,3,4])                          # (N, tilesPerX, 6, T-k+1, (f,p,u))
            Idx = tf.reshape(Idx, [N, -1, 3])                            # (N, tilesPerX*6*(T-k+1), (f,p,u)) transpose and reshape the same way as Z -> (f,p,u) corresponding to Z1 scores
            #print("[DEBUG] >>> Idx reshape:", Idx)
            
            # create tensor that contains for each Z1 score the tile start that needs to be added to the tile position
            T = tf.transpose(posTrack, [1,0,2])                  # shape (N, tilesPerX, (genomeID, contigID, fwdStart, rcStart))
            T = tf.repeat(tf.expand_dims(T, -2), 6, axis=-2)     # shape (N, tilesPerX, 6, (genomeID, contigID, fwdStart, rcStart))
            T = tf.repeat(tf.expand_dims(T, -2), Tk1, axis=-2)   # shape (N, tilesPerX, 6, T-k+1, (genomeID, contigID, fwdStart, rcStart))
            T = tf.reshape(T, [N, -1, 4])                        # shape (N, tilesPerX*6*T-k+1, (genomeID, contigID, fwdStart, rcStart))
            #print("[DEBUG] >>> T:", T)
            
            R = tf.concat((Idx, T), axis=2) # (N, tilesPerX*6*T-k+1, (f,p,u, genomeID, contigID, fwdStart, rcStart))
            #print("[DEBUG] >>> R:", R)
            
            Z1 = tf.reshape(Z1, [-1])            # (N*tilesPerX*6*T-k+1)
            R = tf.reshape(R, [-1, 7])           # (N*tilesPerX*6*T-k+1, 7)
            fwdMask = tf.less(R[:,0], 3)         # (N*tilesPerX*6*T-k+1)
            rcMask = tf.greater_equal(R[:,0], 3) # (N*tilesPerX*6*T-k+1)
            
            Rfwd = tf.boolean_mask(R, fwdMask)
            Zfwd = tf.boolean_mask(Z1, fwdMask)
            #print("[DEBUG] >>> Rfwd:", Rfwd)
            posFwd = tf.multiply(Rfwd[:,1], 3) # rel. pos * 3 -> DNA pos.
            posFwd = tf.add(posFwd, Rfwd[:,0]) # add frame offset
            posFwd = tf.add(posFwd, Rfwd[:,5]) # add tile start
            posFwd = tf.concat([tf.expand_dims(Rfwd[:,3], -1), 
                                tf.expand_dims(posFwd, -1)], axis=1) # (fwdSites, (genomeID, pos))
            
            Rrc = tf.boolean_mask(R, rcMask)
            Zrc = tf.boolean_mask(Z1, rcMask)
            posRC = tf.multiply(Rrc[:,1], 3)     # rel. pos * 3
            posRC = tf.add(posRC, Rrc[:,0])      # add frame offset + 3
            posRC = tf.subtract(posRC, 3)        # correct frame offset
            posRC = tf.subtract(Rrc[:,6], posRC) # subtract pos from reverse tile start
            posRC = tf.subtract(posRC, (k*3)+1)  # go to reverse kmer end
            posRC = tf.concat([tf.expand_dims(Rrc[:,3], -1), 
                               tf.expand_dims(posRC, -1)], axis=1) # (rcSites, (genomeID, pos))
            
            pos = tf.concat((posFwd, posRC), axis=0)
            Z2 = tf.concat((Zfwd, Zrc), axis=0)
            #print("[DEBUG] >>> pos:", pos)
                        
            # store scores in a position matrix
            for i in range(pos.shape[0]):
                n = pos[i,0]
                p = pos[i,1]
                if p >= 0 and p < scores.shape[1]:
                    scores[n,p] = Z2[i]
                #else:
                #    print("[DEBUG] >>> p:", p)
                        
            #print("[DEBUG] >>> scores:", scores)

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
    pxScores = np.empty((N, max([int(np.floor(coordDict[i]['coord'] * len(genomes[i][0]))) for i in range(N)])), dtype=float) # score for each pixel (max binning)
    pxScores[:,:] = np.NaN
    ppxScores = np.array(pxScores, dtype=float)
    rpxScores = np.array(pxScores, dtype=float)
    for g in range(pxScores.shape[0]):
        for j in range(pscores.shape[1]):
            px = int(np.floor(coordDict[g]['coord'] * j))
            if px < pxScores.shape[1]:
                ppxScores[g,px] = np.nanmax([ppxScores[g,px], pscores[g,j]])
                rpxScores[g,px] = np.nanmax([rpxScores[g,px], rscores[g,j]])
            
    minval = np.nanmin([ppxScores, rpxScores])
    maxval = np.nanmax([ppxScores, rpxScores])
    #minval = specProModel.k * (math.log(specProModel.epsilon))
    #maxval = specProModel.k * (math.log(21))
    if minval != maxval:
        assert minval < maxval, str(minval)+" !< "+str(maxval)
        def gradient(val):
            if np.isnan(val):
                return(0,0,0)
            
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
    maxls = np.nanmax(ls)
    minls = np.nanmin(ls)
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
        if not np.isnan(x):
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
    if len(bins) > 1:
        binsize = bins[1] - bins[0]
        p = 1
        while int((p*10)*binsize) != ((p*10)*binsize):
            p += 1

        bins = [np.format_float_positional(b, precision=p) for b in bins]
        
    if ylim is not None:
        assert type(ylim) == tuple, "ylim must be a tuple"
        assert ylim[0] < ylim[1], 'ylim must be a tuple (a, b) where a < b'
        vals = [max(ylim[0], min(ylim[1], y)) for y in vals]
        
    fig, ax = plt.subplots(1, 1, figsize=(16,9))
    ax.bar(x = range(len(bins)),
           height = vals,
           tick_label = bins)
    return fig, ax