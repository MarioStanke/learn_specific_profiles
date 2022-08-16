#!/usr/bin/env python

import argparse
from Bio import SeqIO
import itertools
import json
import numpy as np
import os
import pickle
import tensorflow as tf
from time import time

import sys
sys.path.insert(0, 'modules/')

import sequtils as su
import dataset as dsg
import model
import aadist
import initProfilesExperiment as ipe

sys.path.insert(0, 'modules/GeneLinkDraw/')
import geneLinkDraw as gld

# print if GPU is available
print("", flush=True) # flush: force immediate print, for live monitoring
print("[DEBUG] >>> Tensorflow version", tf.__version__, flush=True)
print("[DEBUG] >>> tf.config.list_physical_devices('GPU'): ", tf.config.list_physical_devices('GPU'), flush=True)
print("[DEBUG] >>> tf.test.is_built_with_gpu_support(): ", tf.test.is_built_with_gpu_support(), flush=True)
print("[DEBUG] >>> tf.test.is_built_with_cuda()", tf.test.is_built_with_cuda(), flush=True)
print("", flush=True)

tfV1, tfV2, _ = tf.__version__.split('.')
assert int(tfV1) >= 2
recentTF = True if int(tfV2) >= 4 else False # from_generator has deprecations from version >= 2.4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse command line arguments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser = argparse.ArgumentParser(description = "Run Grid Search",
                                 formatter_class = argparse.RawTextHelpFormatter)
scriptArgs = parser.add_argument_group("Script Arguments")
scriptArgs.add_argument("--input",
                        dest = "input", 
                        metavar = "FILENAME", 
                        type=argparse.FileType("r"),
                        required=True,
                        help="Filename of the input fasta file (accompanying JSON is found by replacing the extension)")
scriptArgs.add_argument("--output",
                        dest = "output", 
                        metavar = "PATH", 
                        type=argparse.FileType("w"),
                        required=True,
                        help="Path for output file, image is named accordingly")
scriptArgs.add_argument("--grid",
                        dest = "grid", 
                        metavar = "FILENAME", 
                        type=argparse.FileType("r"),
                        required=True,
                        help="Filename of the grid array (JSON)")
scriptArgs.add_argument("--gridID",
                        dest = "gridID", 
                        metavar = "INT", 
                        type=int,
                        required=True)
scriptArgs.add_argument("--font",
                        dest = "font",
                        metavar = "PATH",
                        type = str,
                        default = "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", #"/opt/conda/fonts/Ubuntu-M.ttf",
                        help = "Font to use for link drawing, if file not found drawing is skipped")
args = parser.parse_args()

drawImage = True
if not os.path.exists(args.font):
    print("[WARNING] >>> --font", args.font, "not found, no link image will be drawn")
    drawImage = False



with open(args.grid.name) as fh:
    grid = json.load(fh)

assert args.gridID >= 0, "[ERROR] >>> --gridID "+str(args.gridID)+" must be >= 0"
assert args.gridID < len(grid), "[ERROR] >>> --gridID "+str(args.gridID)+" too big (max "+str(len(grid)-1)+")"

# ~~~~~~~~~~~~
# load genomes
# ~~~~~~~~~~~~

base, ext = os.path.splitext(args.input.name)
jsonname = base+".json"
assert os.path.exists(jsonname), "[ERROR] >>> Could not find "+jsonname

genomes = []
with open(args.input.name, 'rt') as fh:
    for record in SeqIO.parse(fh, "fasta"):
        genomes.append([str(record.seq)])

with open(jsonname, 'rt') as fh:
    posDict = json.load(fh)

#Q = np.ones(21, dtype=np.float32)/21
Q = aadist.getBackgroundDist()

# ~~~~~~~~~~~~~~~~
# general settings
# ~~~~~~~~~~~~~~~~

tile_size = 334  # tile size measured in amino acids
genome_sizes = [sum([len(s) for s in genome]) for genome in genomes]
batch_size = 1  # number of X to generate per batch
tiles_per_X = 13 # number of tiles per X (-> X.shape[0])
steps_per_epoch = max(1, np.mean(genome_sizes) // (batch_size*tiles_per_X*tile_size*3))
lossStrategy = 'experiment'
exp=2
print("[INFO] >>> genome sizes", genome_sizes, " -> ", steps_per_epoch, "steps per epoch")

# unused in model
alpha = 1e-6 # loss norm

# ~~~~~~~~~~~~~~~~~~
# load grid settings
# ~~~~~~~~~~~~~~~~~~

print("[INFO] >>> loading parameters for grid ID", args.gridID)
gridDict = grid[args.gridID]

k = gridDict['k']
s = gridDict['s']
gamma = gridDict['gamma']
l2 = gridDict['l2']
match_score_factor = gridDict['match_score_factor']
learning_rate = gridDict['learning_rate']
profile_plateau_dev = gridDict['profile_plateau_dev']
n_best_profiles = gridDict['n_best_profiles']
mid_factor = gridDict['mid_factor']
bg_factor = gridDict['bg_factor']

# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up and start training
# ~~~~~~~~~~~~~~~~~~~~~~~~~

exhaustiveInitProfiles = ipe.getExhaustiveProfiles(k+(2*s), aadist.getBackgroundDist(), 
                                                   mid_factor=mid_factor, bg_factor=bg_factor, exp=exp)
U = exhaustiveInitProfiles.shape[2]

# build and randomly initialize profile model
tf.keras.backend.clear_session() # avoid memory cluttering by remains of old models
specProModel = model.SpecificProfile(k, su.aa_alphabet_size, U, Q, alpha=alpha, gamma=gamma, l2=l2, shift=s, 
                                     loss=lossStrategy, P_logit_init=exhaustiveInitProfiles)

dsh = dsg.DatasetHelper(genomes, tiles_per_X, tile_size, batch_size, 3)

start = time()
specProModel.train_reporting(genomes, dsh, steps_per_epoch, epochs=500, 
                             learning_rate=learning_rate, profile_plateau=10, profile_plateau_dev=profile_plateau_dev,
                             verbose_freq=10, n_best_profiles=n_best_profiles, match_score_factor=match_score_factor)
end = time()
print(f"[INFO] >>> Training time: {end-start:.2f}")

# get results
P, Pthresh, Ploss = specProModel.getP_report()
Pthresh = np.array(Pthresh) if Pthresh is not None else Pthresh
Ploss = np.array(Ploss) if Ploss is not None else Ploss

# ~~~~~~~~~~~~
# Save history
# ~~~~~~~~~~~~

Phist = P.numpy() if P is not None else None
histdict = {
    'history': specProModel.getHistory(),
    'P_logit': specProModel.P_logit.numpy(),
    'P': Phist,
    'Pthresh': Pthresh,
    'Ploss': Ploss,
    'P_report_raw': specProModel.getP_report_raw().numpy(),
    'P_report_plosshist': specProModel.P_report_plosshist.numpy(),
    'P_report_bestlosshist': [l.numpy() for l in specProModel.P_report_bestlosshist],
    'P_report_bestlosshistIdx': [i.numpy() for i in specProModel.P_report_bestlosshistIdx]
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluate if "genes" were found
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if P is not None:
    # reset genomes to upper case
    for g in range(len(dsh.genomes)):
        for c in range(len(dsh.genomes[g])):
            dsh.genomes[g][c] = dsh.genomes[g][c].upper()

    # get match sites of profiles and create links from them
    thresh = Pthresh
    sites, siteScores, _ = specProModel.get_profile_match_sites(dsh.getDataset(withPosTracking = True), thresh, otherP = P)

    def sitesToLinks(sites, linkThreshold = 100):
        links = []
        skipped = []
        profileToOcc = {}
        linkProfiles = set()
        for g, c, p, u, f in sites:
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
                links.extend(l)
                linkProfiles.add((p, nlinks, str(occs)))

        return links, linkProfiles, skipped



    links, linkProfiles, skipped = sitesToLinks(sites.numpy(), 1000)
    links = [sorted(l) for l in links]

    if drawImage:
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
            
        img, geneToCoord = gld.draw(drawGenes, drawLinks, font = args.font,
                                    genewidth = 20, linkwidth = 1, width = (1920*2))

        imgbase, imgext = os.path.splitext(args.output.name)
        imgname = imgbase+".json"
        img.save(imgname)

    # count links hitting genes
    tp = 0
    noise = 0
    for link in links:
        if all([o[2] >= posDict['start_codon'] and o[2] <= posDict['3flank_start']-k for o in link]):
            tp += 1
        else:
            noise += 1
            
else:
    print("[WARNING] >>> No profiles were reported, no image was drawn")
    tp = 0
    noise = 0



print("[INFO] >>>    TP links:", tp)
print("[INFO] >>> noise links:", noise)

histdict['tp'] = tp
histdict['noise'] = noise

# add .pkl extension if no extension is specified in output
outbase, outext = os.path.splitext(args.output.name)
outname = args.output.name if outext != '' else outbase+".pkl"
with open(outname, 'wb') as fh:
    pickle.dump(histdict, fh)