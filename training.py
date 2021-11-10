#!/usr/bin/env python

import argparse
from Bio import SeqIO
import numpy as np
import pandas as pd
import sequtils as su
import seq
import dataset as dsg
import tensorflow as tf
from time import time

# print if GPU is available
print("", flush=True) # flush: force immediate print, for live monitoring
print("Tensorflow version", tf.__version__, flush=True)
print("tf.config.list_physical_devices('GPU'): ", tf.config.list_physical_devices('GPU'), flush=True)
print("tf.test.is_built_with_gpu_support(): ", tf.test.is_built_with_gpu_support(), flush=True)
print("tf.test.is_built_with_cuda()", tf.test.is_built_with_cuda(), flush=True)
print("", flush=True)

tfV1, tfV2, _ = tf.__version__.split('.')
assert int(tfV1) >= 2
recentTF = True if int(tfV2) >= 4 else False # from_generator has deprecations from version >= 2.4

# ## Parse Command Line Parameters

parser = argparse.ArgumentParser(description = "Run Profile Training",
                                 formatter_class = argparse.RawTextHelpFormatter)
scriptArgs = parser.add_argument_group("Script Arguments")

#scriptArgs.add_argument("--bool",
#                        dest = "",
#                        action = 'store_true',
#                        help = "")

scriptArgs.add_argument("--batch-size",
                        dest = "batchSize", 
                        metavar = "INT", 
                        type=int,
                        default=1,
                        help="Number of X to generate per batch")
scriptArgs.add_argument("--epochs",
                        dest = "epochs", 
                        metavar = "INT", 
                        type=int,
                        default=40,
                        help="Number of epochs to train")
scriptArgs.add_argument("--genome-sizes",
                        dest = "genomeSizes", 
                        metavar = "List of INT", 
                        type=int,
                        nargs="+",
                        default=[10000],
                        help="Sequence lengths of input genomes (bp, ignored if input files are given)")
scriptArgs.add_argument("--input",
                        dest = "input", 
                        metavar = "LIST of FILES", 
                        type=argparse.FileType("r"), 
                        nargs="+",
                        help="Filenames of the input fasta files (if not given, runs on random data)")
scriptArgs.add_argument("--k",
                        dest = "k", 
                        metavar = "INT", 
                        type=int,
                        default=11,
                        help="Profile width (aa)")
scriptArgs.add_argument("--N",
                        dest = "N", 
                        metavar = "INT", 
                        type=int,
                        default=8,
                        help="Number of input genomes (ignored if input files are given)")
scriptArgs.add_argument("--tile-size",
                        dest = "tileSize", 
                        metavar = "INT", 
                        type=int,
                        default=334,
                        help="Cut genome in tiles of this length (aa)")
scriptArgs.add_argument("--tiles-per-X",
                        dest = "tilesPerX", 
                        metavar = "INT", 
                        type=int,
                        default=13,
                        help="Number of tiles per training step")
scriptArgs.add_argument("--U",
                        dest = "U", 
                        metavar = "INT", 
                        type=int,
                        default=2000,
                        help="Number of profiles to train")

args = parser.parse_args()

#print("[DEBUG] >>>", args)

if args.input is not None:
    useRealData = True
    input = [f.name for f in args.input]
    assert len(input) > 0, "No input genome"

else:
    useRealData = False
    input = None
    assert args.N > 0, "Specify at least one random genome"
    assert len(args.genomeSizes) > 0, "Specify at least one random sequence length"
    for l in args.genomeSizes:
        assert l > 0, "Random sequence lengths must be >0"

assert args.batchSize > 0, "batch-size must be >0"
assert args.epochs > 0, "epochs must be >0"
assert args.k > 0, "k must be >0"
assert args.tileSize > 0, "tile-size must be >0"
assert args.tilesPerX > 0, "tiles-per-X must be >0"
assert args.U > 0, "U must be >0"



# ## Prepare Training

if useRealData:
    genomes = [[] for _ in range(len(input))]
    seqnames = [[] for _ in range(len(input))]
    
    for i in range(len(input)):
        genomes[i].extend([str(seq.seq) for seq in SeqIO.parse(input[i], 'fasta')])
        seqnames[i].extend([str(seq.id) for seq in SeqIO.parse(input[i], 'fasta')])

else:
    seqnames = None
    genomeSizes = [args.genomeSizes] * args.N
    print("[DEBUG] >>> genomeSizes:", genomeSizes)
    insertPatterns = ["ATGGCAAGAATTCAATCTACTGCAAATAAAGAA"] # ['MARIQSTANKE', 'WQEFNLLQIK', 'GKNSIYCK*R', 'FFICSRLNSCH', 'SLFAVD*ILA', 'LYLQ*IEFLP']
    repeatPatterns = ['AGAGAACCTGAAGCTACTGCTGAACCTGAAAGA'] # ['REPEATAEPER', 'ENLKLLLNLK', 'RT*SYC*T*K', 'SFRFSSSFRFS', 'LSGSAVASGS', 'FQVQQ*LQVL']
    genomes, repeatTracking, insertTracking = seq.getRandomGenomes(args.N, genomeSizes, 
                                                                   insertPatterns,
                                                                   repeatPatterns,
                                                                   mutationProb=0.0, 
                                                                   repeatMultiple=range(0,1),
                                                                   repeatInsert=range(10,11),
                                                                   verbose=False)

#Q = seq.backGroundAAFreqs(genomes, True)
Q = np.ones(21, dtype=np.float32)/21 # uniform background distribution appears to be rather better

genomeSizeSums = [sum([len(s) for s in genome]) for genome in genomes]
stepsPerEpoch = max(1, np.mean(genomeSizeSums) // (args.batchSize*args.tilesPerX*args.tileSize*3))
print("[DEBUG] >>> genomeSizeSums -> stepsPerEpoch:", genomeSizeSums, " -> ", stepsPerEpoch)

def getDataset(tilesPerX: int = args.tilesPerX,
               tileSize: int = args.tileSize,
               genomes = genomes,
               withPosTracking: bool = False):
    if recentTF:
        if withPosTracking:
            ds = tf.data.Dataset.from_generator(
                dsg.createBatch,
                args = (tf.constant(tilesPerX), tf.constant(tileSize), tf.constant(genomes, dtype=tf.string), 
                        tf.constant(True)),
                output_signature = (tf.TensorSpec(shape = ([tilesPerX, len(genomes), 6, tileSize, su.aa_alphabet_size], 
                                                        [tilesPerX, len(genomes), 3]),
                                                dtype = (tf.float32, tf.int32)))
            )
        else:
            ds = tf.data.Dataset.from_generator(
                dsg.createBatch,
                args = (tf.constant(tilesPerX), tf.constant(tileSize), tf.constant(genomes, dtype=tf.string), 
                        tf.constant(False)),
                output_signature = (tf.TensorSpec(shape = [tilesPerX, len(genomes), 6, tileSize, su.aa_alphabet_size],
                                                dtype = tf.float32))
            )
    else: # uses deprecated arguments
        if withPosTracking:
            ds = tf.data.Dataset.from_generator(
                dsg.createBatch,
                args = (tf.constant(tilesPerX), tf.constant(tileSize), tf.constant(genomes, dtype=tf.string), 
                        tf.constant(True)),
                output_types = (tf.float32, tf.int32),
                output_shapes = (tf.TensorShape([tilesPerX, len(genomes), 6, tileSize, su.aa_alphabet_size]),
                                 tf.TensorShape([tilesPerX, len(genomes), 3]))
            )
        else:
            ds = tf.data.Dataset.from_generator(
                dsg.createBatch,
                args = (tf.constant(tilesPerX), tf.constant(tileSize), tf.constant(genomes, dtype=tf.string), 
                        tf.constant(False)),
                output_types = (tf.float32),
                output_shapes = (tf.TensorShape([tilesPerX, len(genomes), 6, tileSize, su.aa_alphabet_size]))
            )

    return ds



# ## Define Model

class SpecificProfile(tf.keras.Model):
    def __init__(self, k, alphabet_size, units, Q, P_logit_init=None, **kwargs):
        super().__init__(**kwargs)
        # P_logit_init = tf.random.normal([k, alphabet_size, units], stddev=.5, dtype=tf.float32, seed=1)
        
        self.Q = Q
        self.k = k
        self.alphabet_size = alphabet_size
        self.units = units
        self.history = {'loss': [],
                        'Rmax': [],
                        'Rmin': [],
                        'Smax': [],
                        'Smin': []}
        
        if P_logit_init is None:
            P_logit_init = self._getRandomProfiles()
        if False: # to test whether the right pattern has low loss
                P_logit_init[:,:,0:2] = -100. #-100 *np.ones([k, 21, 2], dtype=np.float32)
                P_logit_init[0,15,0] = 5 # M
                P_logit_init[1,7,0] = 5 # A
                P_logit_init[2,16,0] = 5 # R
                P_logit_init[3,8,0] = 5 # I
                P_logit_init[4,19,0] = 5 # Q
                P_logit_init[5,12,0] = 5 # S
                P_logit_init[6,4,0] = 5 # T
                P_logit_init[7,7,0] = 5 # A
                P_logit_init[8,9,0] = 5 # N
                P_logit_init[9,1,0] = 5 # K
                P_logit_init[10,2,0] = 5 # E
                # REPEATAEPER
                P_logit_init[0,16,1] = 5 # R
                P_logit_init[1,2,1] = 5 # E
                P_logit_init[2,18,1] = 5 # P
                P_logit_init[3,2,1] = 5 # E
                P_logit_init[4,7,1] = 5 # A
                P_logit_init[5,4,1] = 5 # T
                P_logit_init[6,7,1] = 5 # A
                P_logit_init[7,2,1] = 5 # E
                P_logit_init[8,18,1] = 5 # P
                P_logit_init[9,2,1] = 5 # E
                P_logit_init[10,16,1] = 5 # R
                
        self.setP_logit(P_logit_init)
        
    def _getRandomProfiles(self):
        Q1 = tf.expand_dims(self.Q,0)
        Q2 = tf.expand_dims(Q1,-1)
        
        P_logit_like_Q = np.log(Q2.numpy())
        P_logit_init = P_logit_like_Q + np.random.normal(scale=4., size=[self.k, self.alphabet_size, self.units]).astype('float32')
        return P_logit_init
        
    # return for each profile the best score at any position in the dataset
    def get_profile_match_sites(self, ds, threshold, aa_tile_size, genomes, L5score: bool = False):
        # dict of dicts of dict, for each genome, map each contig to a dict that collects profile indices and positions
        #   (given that the profile matches with score above threshold for that genome, contig and position)
        sites = {}
        for batch in ds:
            assert len(batch) == 2, str(len(batch))+" -- use batch dataset with position tracking!"
            X = batch[0]        # (B, tilePerX, N, 6, tileSize, 21)
            posTrack = batch[1] # (B, tilePerX, N, 3)
            assert len(X.shape) == 6, str(X.shape)
            assert len(posTrack.shape) == 4, str(posTrack.shape)
            assert X.shape[0:3] == posTrack.shape[0:3], str(X.shape)+" != "+str(posTrack.shape)
            for b in range(X.shape[0]): # iterate samples in batch
                # get positions of best profile matches
                _, _, Z = self.call(X[b])              # (tilePerX, N, 6, T-k+1, U)
                if L5score:
                    gamma = .2
                    Z2 = tf.nn.softmax(gamma*Z, axis=0)
                    Z3 = tf.math.multiply(Z, tf.square(Z2))
                    Z = Z3
                    
                for g in range(X.shape[2]):
                    if tf.reduce_max(Z[:,g,:,:,:]) >= threshold:
                        p = tf.argmax( tf.reduce_max(Z[:,g,:,:,:], axis=[0,1,2]) ).numpy() # index of the profile with best hit
                        t = tf.argmax( tf.reduce_max(Z[:,g,:,:,p], axis=[1,2]) ).numpy()   # index of the tile with best hit
                        f = tf.argmax( tf.reduce_max(Z[t,g,:,:,p], axis=[1]) ).numpy()     # index of the frame with best hit
                        r = tf.argmax( Z[t,g,f,:,p] ).numpy()                              # index of the rel. position with best hit

                        c = posTrack[b,t,g,0].numpy()
                        if c < 0:
                            continue # either bug or profile matches good on padding
                            
                        assert c < len(genomes[g]), str(c)+" >= "+str(len(genomes[g]))+" for genome "+str(g)
                            
                        cStart = posTrack[b,t,g,1].numpy()
                        tilelen = posTrack[b,t,g,2].numpy()
                        
                        pos = r*3 # to dna coord
                        if f < 3:
                            pos += f # add frame shift
                            pos = cStart + pos
                        else:
                            # add appropriate frame shift
                            seqlen = len(genomes[g][c])
                            if seqlen%3 == tilelen%3:
                                pos += f-3
                            else:
                                pos += dsg.rcFrameOffsets(seqlen)[f-3]
                                
                            pos = tilelen - pos - (self.k*3)

                        if g not in sites:
                            sites[g] = {}
                        if c not in sites[g]:
                            sites[g][c] = {'profile': [], 'pos': [], 'score': [], 'frame': []}

                        sites[g][c]['profile'].append(p)
                        sites[g][c]['pos'].append(pos)
                        sites[g][c]['score'].append(Z[t,g,f,r,p].numpy())
                        sites[g][c]['frame'].append(f)
                        
        return sites

    def getP(self):
        P = tf.nn.softmax(self.P_logit, axis=1, name="P")
        return P
    
    def getR(self):
        P = self.getP()
        Q1 = tf.expand_dims(self.Q,0)
        Q2 = tf.expand_dims(Q1,-1)
        # Limit the odds-ratio, to prevent problem with log(0).
        # Very bad matches of profiles are irrelevant anyways.
        ratio = tf.maximum(P/Q2, 1e-6)
        R = tf.math.log(ratio)
        return R
    
    def getZ(self, X):
        R = self.getR()

        X1 = tf.expand_dims(X,-1) # 1 input channel
        R1 = tf.expand_dims(R,-2) # 1 input channel
        Z1 = tf.nn.conv2d(X1, R1, strides=1,
                          padding='VALID', data_format="NHWC", name="Z")
        Z = tf.squeeze(Z1, 4) # remove input channel dimension
        return Z, R
        
    def call(self, X):
        Z, R = self.getZ(X)

        S = tf.reduce_max(Z, axis=[2,3])
        return S, R, Z

    # custom loss
    def loss(self, S):
        # overall fit of patterns
        # beta = 5.0 # the larger the harder the softmax
        # S1 = tf.math.multiply(S, tf.nn.softmax(beta*S, axis=0)) # a soft version between maximizing and summing over the tiles
        L1 = -tf.reduce_sum(S) / self.units
        
        # homogeneity along pattern
        P = self.getP()
        H = -tf.reduce_sum(tf.math.multiply(P, tf.math.log(P)), axis=1) # entropy
        # standard deviation of entropies
        VH = tf.math.reduce_std(H, axis=0) # variance for each profile along site axis
        L2 = tf.reduce_mean(VH)
        
        # homogeneity between genomes
        # ignore the best score per genome, so that rather others are improved
        # L3 = tf.reduce_sum(tf.reduce_max(S, axis=1)) # cancels out the best occurence from L1
        
        # other ideas: 
        # - std deviation of scores of different genomes
        # L4 = tf.reduce_sum(tf.math.reduce_std(tf.reduce_max(S, axis=0), axis=0))
        # - minimum score of any genome
        
        # penalize multiple similarly good near-best matches in the same genome
        gamma = .2 # a small value means a more inclusive meaning of near-best
        S2 = tf.nn.softmax(gamma*S, axis=0)
        S3 = tf.reduce_max(S2, axis=0) # the closer to 1, the clearer is the champion match a winner
        S4 = tf.reduce_max(S, axis=0) # ranges over tiles, or soft max like in L1
        S5 = tf.math.multiply(S4, tf.square(S3)) # effectively the best score per genome is divided by the number of matches
        loss_by_unit = tf.reduce_sum(-S5, axis=0) / self.units # sum over genomes
        L5 = tf.reduce_sum(loss_by_unit) # sum over profiles=units
        
        #return (L1+L3)/N + 50*L2, L2, L4 # + 100*L2 # (L1 + L3)/N #+ 100*L2
        return L5, loss_by_unit, (L1, L2, (L5, loss_by_unit)) # first: loss to use in training, last: tuple of all losses for evaluation

    # return for each profile the best score at any position in the dataset
    def max_profile_scores(self, ds):
        scores = np.ones([self.units], dtype=np.float32) * -np.infty
        for batch in ds:
            assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                S, _, _ = self.call(X)
                scores = np.maximum(np.max(S, axis=(0,1)), scores)
                                    
        return scores
    
    # return for each profile the loss contribution
    def min_profile_losses(self, ds):
        losses = np.zeros([self.units], dtype=np.float32)
        for batch in ds:
            assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                S, _, _ = self.call(X)
                _, losses_by_unit, _ = self.loss(S)
                losses += losses_by_unit
                                    
        return losses
    
    @tf.function()
    def train_step(self, X):
        with tf.GradientTape() as tape:
            S, R, _ = self.call(X)
            L, _, _ = self.loss(S)

        grad = tape.gradient(L, self.P_logit)
        self.opt.apply_gradients([(grad, self.P_logit)])
        
        return S, R, L

    def train(self, X, epochs=1000, verbose=True):
        self.opt = tf.keras.optimizers.Adam(learning_rate=.1) # large learning rate is much faster
        for i in range(epochs):
            self.train_step(X)
            if verbose and (i%(100) == 0 or i==epochs-1):
                S, R, _ = self(X)
                L, _, _ = self.loss(S)
                print(f"epoch {i:>5} loss={L.numpy():.4f}" +
                      " max R: {:.3f}".format(tf.reduce_max(R).numpy()) +
                      " min R: {:.3f}".format((tf.reduce_min(R).numpy())))
                
    def train_ds(self, ds, steps_per_epoch, epochs, verbose=True, verbose_freq=100):
        self.opt = tf.keras.optimizers.Adam(learning_rate=1.) # large learning rate is much faster
        tstart = time()
        for i in range(epochs):
            steps = 0
            Lb = []
            Smin, Smax = float('inf'), float('-inf')
            Rmin, Rmax = float('inf'), float('-inf')
            for batch in ds:
                assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
                for X in batch:
                    assert len(X.shape) == 5, str(X.shape)
                    S, R, L = self.train_step(X)
                    Lb.append(L)
                    Rmax = max(Rmax, tf.reduce_max(R).numpy())
                    Rmin = min(Rmin, tf.reduce_min(R).numpy())
                    Smax = max(Smax, tf.reduce_max(S).numpy())
                    Smin = min(Smin, tf.reduce_min(S).numpy())
                    
                steps += 1
                if steps >= steps_per_epoch:
                    break
                    
            self.history['loss'].append(np.mean(Lb))
            self.history['Rmax'].append(Rmax)
            self.history['Rmin'].append(Rmin)
            self.history['Smax'].append(Smax)
            self.history['Smin'].append(Smin)
                    
            if verbose and (i%(verbose_freq) == 0 or i==epochs-1):
                S, R, _ = self(X)
                L, _, _ = self.loss(S)
                tnow = time()
                print(f"epoch {i:>5} loss={L.numpy():.4f}" +
                      " max R: {:.3f}".format(tf.reduce_max(R).numpy()) +
                      " min R: {:.3f}".format((tf.reduce_min(R).numpy())) +
                      " time: {:.2f}".format(tnow-tstart)) 
                
    def seed_P_ds(self, ds):
        """
            Seed profiles P with profiles that represent units random positions in the input sequences.
            Positions are drawn uniformly from all positions in all sequences. 
            This is done with an old and neat trick online, so that the data has to be read only once.
        """
        rho = 2.0
        oneProfile_logit_like_Q = np.log(self.Q)
        P_logit_init = self._getRandomProfiles() # shape [k, alphabet_size, units]
        m = 0 # number of positions seen so far
        for batch in ds:
            assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                X = X.numpy()
                PP = X.reshape([-1, self.alphabet_size]) # (tilesPerX, N, 6, T, alphSize) -> (tilesPerX*N*6*T, alphSize)
                J = PP.shape[0]
                # PP[j,a] is 1 if the j-th character in an artificially concatenated sequence is char a
                # the length k patterns extend over tile ends, which could be improved later
                num_pos = J - self.k
                for j in range(num_pos):
                    i = -1 # i-th profile is to be replaced, unless i<0
                    if m < self.units:
                        i = m
                    elif np.random.choice(m) < self.units:
                        i = np.random.choice(self.units)
                    if i >= 0:
                        # replace i-th profile with a seed profile build from the pattern starting at position j
                        # Seed is the background distribution, except the observed k-mer at pos j is more likely
                        seed = rho * PP[j:j+self.k,:] + oneProfile_logit_like_Q
                        P_logit_init[:,:,i] = seed
                        # print (f"replacing {i:>3}-th pattern with the one from pos {j:>6}:\n", seed)
                    m += 1

        return P_logit_init
                
    def setP_logit(self, P_logit_init):
        self.P_logit = tf.Variable(P_logit_init, trainable=True, name="P_logit") 



# ## Start Training

specProModel = SpecificProfile(args.k, su.aa_alphabet_size, args.U, Q)
ds = getDataset().repeat().batch(args.batchSize).prefetch(150)
#ds_score = getDataset().batch(args.batchSize).prefetch(150)
ds_init  = getDataset().batch(args.batchSize).prefetch(150)

P_logit_init = specProModel.seed_P_ds(ds_init)
specProModel.setP_logit(P_logit_init)

start = time()
specProModel.train_ds(ds, stepsPerEpoch, args.epochs, verbose_freq=np.math.ceil(args.epochs/20))
end = time()
print(f"[DEBUG] >>> training time: {end-start:.2f}")
print("[DEBUG] >>> History:\n", pd.DataFrame(specProModel.history))