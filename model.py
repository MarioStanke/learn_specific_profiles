#!/usr/bin/env python

from operator import pos
import numpy as np
import dataset as dsg
import tensorflow as tf
from time import time

from tensorflow.python.ops.gen_array_ops import shape



# Helper Functions for Profile Cleanup
def cosineSim(a: np.ndarray, b: np.ndarray):
    assert len(a) == len(b)
    return np.sum(a*b) / (np.sqrt(np.sum(np.square(a))) * np.sqrt(np.sum(np.square(b))))

# modified version of https://gist.github.com/nornagon/6326a643fc30339ece3021013ed9b48c
# enforce gapless alignment
def smith_waterman(a, b) -> float:
    assert a.shape[1] == b.shape[1], str(a.shape)+", "+str(b.shape)
    # H holds the alignment score at each point, computed incrementally
    H = np.zeros((a.shape[0] + 1, b.shape[0] + 1))
    for i in range(1, a.shape[0] + 1):
        for j in range(1, b.shape[0] + 1):
            # The score for substituting the letter a[i-1] for b[j-1]. Generally low
            # for mismatch, high for match.
            H[i,j] = H[i-1,j-1] + cosineSim(a[i-1], b[j-1])
    # The highest score is the best local alignment.
    # For our purposes, we don't actually care _what_ the alignment was, just how
    # aligned the two strings were.
    return H.max()

# create index tensor for a data batch of shape (T, N, F, P, U)
#   returns tensor of shape (T, N, F, P, U, 3) where each element 
#   of the second-last dimension is a list with indices [f,p,u] (frame, rel.pos. profile)
def indexTensor(T, N, F, P, U):
    Uidx = tf.broadcast_to(tf.range(0, U, dtype=tf.int32), (T,N,F,P,U)) # (T, N, 6, P, U)
    assert Uidx.shape == [T,N,F,P,U], str(Uidx.shape)
    Pidx = tf.broadcast_to(tf.range(0, P, dtype=tf.int32), (T,N,F,P))   # (T, N, 6, P)
    Pidx = tf.repeat(tf.expand_dims(Pidx, -1), [U], axis=-1)            # (T, N, 6, P, U)
    assert Pidx.shape == [T,N,F,P,U], str(Pidx.shape)
    Fidx = tf.broadcast_to(tf.range(0, F, dtype=tf.int32), (T,N,F))     # (T, N, 6)
    Fidx = tf.repeat(tf.expand_dims(Fidx, -1), [P], axis=-1)            # (T, N, 6, P)
    Fidx = tf.repeat(tf.expand_dims(Fidx, -1), [U], axis=-1)            # (T, N, 6, P, U)
    assert Fidx.shape == [T,N,F,P,U], str(Fidx.shape)
    I = tf.stack((Fidx, Pidx, Uidx), axis=5)
    assert I.shape == [T,N,F,P,U,3], str(I.shape)
    return I



class SpecificProfile(tf.keras.Model):
    def __init__(self, k, alphabet_size, units, Q, P_logit_init=None, alpha=1e-6, gamma=0.2, shift=0, **kwargs):
        super().__init__(**kwargs)
        
        self.Q = Q                         # shape: (alphabet_size)
        self.alpha = alpha
        self.gamma = gamma # a small value means a more inclusive meaning of near-best ([3, 4, 1] -> [0.25, 0.7, 0.05] // [.3, .4, .1] -> [0.34, 0.38, 0.28])
        #perfect_match_score = np.log(1/np.mean(Q))  # set epsilon to a value that mismatch score is roughly -1*perfect match score
        #self.epsilon = np.exp(-perfect_match_score) # original value was 1e-6, which led to mismatch score of -13.8 while perfect match score is only 3
        self.epsilon = 1e-6
        
        self.k = k                         # shape: ()
        self.s = shift                     # shape: ()
        self.alphabet_size = alphabet_size # shape: ()
        self.units = units                 # shape: ()
        self.history = {'loss': [],
                        'Rmax': [],
                        'Rmin': [],
                        'Smax': [],
                        'Smin': []}
        
        if P_logit_init is None:
            P_logit_init = self._getRandomProfiles()
                
        self.setP_logit(P_logit_init)      # shape: (k, alphabet_size, U)
        self.P_report = []
        self.P_report_whole = []
        self.P_report_thresold = []
        self.P_report_loss = []
        self.P_report_masked_sites = []
        self.P_report_kmer_scores = []
        
    def _getRandomProfiles(self):
        Q1 = tf.expand_dims(self.Q,0)
        Q2 = tf.expand_dims(Q1,-1)         # shape: (1, alphabet_size, 1)
        
        P_logit_like_Q = np.log(Q2.numpy())
        P_logit_init = P_logit_like_Q + np.random.normal(scale=4., size=[self.k+(2*self.s), self.alphabet_size, self.units]).astype('float32')
        return P_logit_init                # shape: (k, alphabet_size, U)
        
    # return for each (or one desired) profile match positions in the dataset, given a score threshold
    def get_profile_match_sites(self, ds, score_threshold, pIdx = None, otherP = None):
        matches = None
        scores = None
        for batch in ds:
            #assert len(batch) == 2, str(len(batch))+" -- use batch dataset with position tracking!"
            X_b = batch[0]        # (B, tilePerX, N, 6, tileSize, 21)
            posTrack_b = batch[1] # (B, tilePerX, N, 6, 4)
            assert len(X_b.shape) == 6, str(X_b.shape)
            assert posTrack_b.shape != (1, 0), str(posTrack.shape)+" -- use batch dataset with position tracking!"
            assert X_b.shape[0:4] == posTrack_b.shape[0:4], str(X_b.shape)+" != "+str(posTrack_b.shape)
            pTdim = posTrack_b.shape[-1]
            for b in range(X_b.shape[0]): # iterate samples in batch
                X = X_b[b]
                posTrack = posTrack_b[b]                                           # (tilePerX, N, 6, (genomeID, contigID, startPos, aa_seqlen))
                _, _, Z = self.call(X, otherP)                                     # (tilePerX, N, 6, T-k+1, U)
                if pIdx is not None:
                    Z = Z[:,:,:,:,pIdx:(pIdx+1)] # only single profile, but keep dimensions
                
                Idx = indexTensor(Z.shape[0], Z.shape[1], Z.shape[2],
                                  Z.shape[3], Z.shape[4])                          # (tilePerX, N, 6, T-k+1, U, (f,r,u)) (f - frame, r - rel. pos., u - profile)

                # collapse genome and tile dimensions
                Z = tf.reshape(Z, [-1, Z.shape[-3], Z.shape[-2], Z.shape[-1]])     # (tilesPerX*N, 6, T-k+1, U)
                M = tf.greater_equal(Z, score_threshold)                           # (tilesPerX*N, 6, T-k+1, U), >>> consider match if score >= threshold <<<
                T = tf.reshape(posTrack, [-1, 6, pTdim])                           # (tilesPerX*N, 6, (genomeID, contigID, startPos, aa_seqlen))
                Idx = tf.reshape(Idx, [-1, Idx.shape[-4], Idx.shape[-3], 
                                           Idx.shape[-2], Idx.shape[-1]])          # (tilesPerX*N, 6, T-k+1, U, (f,r,u))
                
                # reduce to genome tiles that have matches
                Mgentile = tf.reduce_any(M, axis=[2,3])     # (tilesPerX*N, 6)
                Mgentile = tf.logical_and(Mgentile, tf.not_equal(T[:,:,0], -1)) # also set exhausted contigs to False
                Mgentile = tf.reduce_any(Mgentile, axis=1)  # (tilesPerX*N), of which `matches` are True
                T = tf.boolean_mask(T,Mgentile)             # (matches, 6, (genomeIDs, contigIDs, startPos, aa_seqlen))
                M = tf.boolean_mask(M,Mgentile)             # (matches, 6, T-k+1, U)
                Z = tf.boolean_mask(Z,Mgentile)             # (matches, 6, T-k+1, U)
                Idx = tf.boolean_mask(Idx, Mgentile)        # (matches, 6, T-k+1, U, 3)

                # manual broadcast of T to the correct shape
                T = tf.repeat(tf.expand_dims(T, 2), [M.shape[-1]], axis=2) # (matches, 6, U, (genomeID, contigID, startPos, aa_seqlen))
                T = tf.repeat(tf.expand_dims(T, 2), [M.shape[-2]], axis=2) # (matches, 6, T-k+1, U, (genomeID, contigID, startPos, aa_seqlen))
                #T = tf.repeat(tf.expand_dims(T, 1), [M.shape[-3]], axis=1) # (matches, 6, T-k+1, U, (genomeID, contigID, fwdStart, rcStart))
                
                # reduce to single match sites
                Idx = tf.boolean_mask(Idx, M)    # (sites, (f, r, u))
                T = tf.boolean_mask(T, M)        # (sites, (genomeID, contigID, startPos, aa_seqlen))
                Z = tf.boolean_mask(Z, M)        # (sites) where each entry is a score
                Z = tf.expand_dims(Z, -1)        # (sites, (score))
                R = tf.concat((Idx, T), axis=1 ) # (sites, (f, r, u, genomeID, contigID, startPos, aa_seqlen))

                # continue with vectorized operations, tf.map_fn to dsg.restoreGenomePosition is super slow even with parallel threads enabled
                fwdMask = tf.less(R[:,0], 3)         # (sites)
                rcMask = tf.greater_equal(R[:,0], 3) # (sites)
                rFwd = tf.boolean_mask(R, fwdMask)   # (fwdSites, (f, r, u, genomeID, contigID, startPos, aa_seqlen, score))
                rRC = tf.boolean_mask(R, rcMask)     # ( rcSites, (f, r, u, genomeID, contigID, startPos, aa_seqlen, score))
                zFwd = tf.boolean_mask(Z, fwdMask)   # (fwdSites, (score))
                zRC = tf.boolean_mask(Z, rcMask)     # ( rcSites, (score))

                # fwd case: p *= 3; p += tileStart
                posFwd = tf.multiply(rFwd[:,1], 3)             # (fwdSites), *3
                #posFwd = tf.add(posFwd, rFwd[:,0])             # (fwdSites), add frame offset
                posFwd = tf.add(posFwd, rFwd[:,5])             # (fwdSites), add start
                sites = tf.concat([tf.expand_dims(rFwd[:,3], -1), 
                                   tf.expand_dims(rFwd[:,4], -1), 
                                   tf.expand_dims(posFwd, -1), 
                                   tf.expand_dims(rFwd[:,2], -1), 
                                   tf.expand_dims(rFwd[:,0], -1)], axis=1) # (fwdSites, (genomeID, contigID, pos, u, f))
                
                # rc case: p *= 3; p = rcStart - p - (k*3) + 1
                posRC = tf.multiply(rRC[:,1], 3)               # (rcSites), *3
                #posRC = tf.add(posRC, rRC[:,0])                # (rcSites), add frame offset
                #posRC = tf.subtract(posRC, 3)                  # (rcSites), -3
                posRC = tf.subtract(rRC[:,5], posRC)           # (rcSites), start - p
                posRC = tf.subtract(posRC, (self.k*3))         # (rcSites), -(k*3)
                posRC = tf.add(posRC, 1)                       # (rcSites), +1
                sitesRC = tf.concat([tf.expand_dims(rRC[:,3], -1), 
                                     tf.expand_dims(rRC[:,4], -1), 
                                     tf.expand_dims(posRC, -1), 
                                     tf.expand_dims(rRC[:,2], -1), 
                                     tf.expand_dims(rRC[:,0], -1)], axis=1) # (fwdSites, (genomeID, contigID, pos, u, f))
                
                sites = tf.concat((sites, sitesRC), axis=0) # (sites, (genomeID, contigID, pos, u, f))
                siteScores = tf.concat((zFwd, zRC), axis=0) # (sites, (score))
                if matches is None:
                    matches = sites
                    scores = siteScores
                else:
                    matches = tf.concat([matches, sites], axis=0)
                    scores = tf.concat([scores, siteScores], axis=0)

        if matches is None:
            matches = tf.constant([])
            scores = tf.constant([])
        
        return matches, scores

    
    # return index of the profile that has the lowest loss
    def get_best_profile(self, ds):
        Ls = []
        for batch in ds:
            X = batch[0]        # (B, tilePerX, N, 6, tileSize, 21)
            posTrack = batch[1] # (B, tilePerX, N, 6, 4)
            assert len(X.shape) == 6, str(X.shape)
            assert posTrack.shape != (1, 0), str(posTrack.shape)+" -- use batch dataset with position tracking!"
            assert X.shape[0:4] == posTrack.shape[0:4], str(X.shape)+" != "+str(posTrack.shape)
            for b in range(X.shape[0]): # iterate samples in batch
                _, _, Z = self.call(X[b])               # Z: (ntiles, N, 6, tile_size-k+1, U)
                #Zl = self._loss_calculation(Z)          #    (N, U, ntiles*6*(tile_size-k+1))
                #L = tf.reduce_mean(Zl, axis=[0,2])      #                                 (U), mean score of each profile
                _, L = self.loss(Z)                     # (U)

                W = tf.cast(posTrack[b,:,:,:,0] != -1, tf.float32) # (tilePerX, N, f) -> -1 if contig was exhausted -> False if exhausted -> 1 for valid contig, 0 else
                W = tf.multiply(tf.reduce_sum(W), tf.ones(shape = [self.units], dtype=tf.float32)) # weight for the means, shape (U)
                Ls.append(tf.multiply(L, W).numpy()) # store weighted losses

                if tf.reduce_any( tf.math.is_nan(Z) ):
                    print("[DEBUG] >>> nan in Z")
                    print("[DEBUG] >>> M:", M)
                    print("[DEBUG] >>> W:", W)
                    print("[DEBUG] >>> W1:", W1)
                    print("[DEBUG] >>> Ms:", Ms)

        B = tf.reduce_mean(Ls, axis=0) # get overall lowest mean loss per profile
        return tf.argmin(B), tf.reduce_min(B) # index, mean loss

    def getP(self):
        P = tf.nn.softmax(self.P_logit, axis=1, name="P")
        return P                                 # shape: (k, alphabet_size, U)

    def getP_report_raw(self):
        P = tf.transpose(self.P_report, [1,2,0]) # shape: (k, alphabet_size, -1)
        return P
    
    def getP_report(self):
        P = tf.nn.softmax(self.getP_report_raw(), axis=1, name="P_report")
        return P, self.P_report_thresold, self.P_report_loss
    
    def getP_report_whole(self):
        P = tf.transpose(self.P_report_whole, [1,2,0]) # shape: (k, alphabet_size, -1)
        P = tf.nn.softmax(P, axis=1, name="P_report_whole")
        return P
    
    def getR(self, otherP = None):
        P = self.getP() if otherP is None else otherP
        Q1 = tf.expand_dims(self.Q,0)
        Q2 = tf.expand_dims(Q1,-1)
        # Limit the odds-ratio, to prevent problem with log(0).
        # Very bad matches of profiles are irrelevant anyways.
        ratio = tf.maximum(P/Q2, self.epsilon)
        R = tf.math.log(ratio)
        if tf.reduce_any(tf.math.is_nan(P)):
            print("[DEBUG] >>> nan in P:", tf.reduce_any(tf.math.is_nan(P), axis=[0,1]), 
                  tf.boolean_mask(P, tf.reduce_any(tf.math.is_nan(P), axis=[0,1]), axis=2))
            print("[DEBUG] >>> Q:", self.Q)
            
        return R                                 # shape: (k, alphabet_size, U)
    
    def getZ(self, X, otherP = None):
        R = self.getR(otherP)

        X1 = tf.expand_dims(X,-1) # 1 input channel   shape: (ntiles, N, 6, tile_size, alphabet_size, 1)
        R1 = tf.expand_dims(R,-2) # 1 input channel   shape: (k, alphabet_size, 1, U)

        # X1: (batch_shape (ntiles, N, 6), in_height (tile_size),     in_width (alphabet_size), in_channels (1))
        # R1:                                 (filter_height (k), filter_width (alphabet_size), in_channels (1), out_channels (U))
        # Z1: (batch_shape (ntiles, N, 6), tile_size-k+1, 1, U)
        Z1 = tf.nn.conv2d(X1, R1, strides=1,
                          padding='VALID', data_format="NHWC", name="Z")
        Z = tf.squeeze(Z1, 4) # remove input channel dimension   shape (ntiles, N, 6, tile_size-k+1, U)
        
        if tf.reduce_any(tf.math.is_nan(R)):
            print("[DEBUG] >>> nan in R")
        if tf.reduce_any(tf.math.is_nan(X)):
            print("[DEBUG] >>> nan in X")
        if tf.reduce_any(tf.math.is_nan(Z)):
            print("[DEBUG] >>> nan in Z")
        
        return Z, R
        
    def call(self, X, otherP = None):
        Z, R = self.getZ(X, otherP)

        S = tf.reduce_max(Z, axis=[2,3])   # shape (ntiles, N, U)
        return S, R, Z

    # Mario's loss
    def loss(self, Z):
        #print ("1st of loss calculation: Z.shape=", Z.shape)
        # shape of Z: ntiles x N x 6 x tile_size-k+1 x U 
        S = tf.reduce_max(Z, axis=[0,2,3]) # N x U
        score = tf.reduce_sum(S)
        
        Z = tf.transpose(Z, [1,4,0,2,3]) # shape N x U x ntiles x 6 x tile_size-k+1
        # print ("2nd of loss calculation: Z.shape=", Z.shape)
        Z = tf.reshape(Z, [Z.shape[0], Z.shape[1], -1])
        # print ("3rd of loss calculation: Z.shape=", Z.shape, "\n", tf.reduce_max(Z, axis=[-1]))
        Zsm = tf.nn.softmax(Z, axis=-1) # compute softmax
        Z = tf.math.multiply(Z, Zsm)
        # print ("4th of loss calculation: Z.shape=", Z.shape, "\n", tf.reduce_max(Z, axis=[-1]))
        loss_by_unit = -tf.math.reduce_max(Z, axis=-1)
        loss_by_unit = tf.reduce_sum(loss_by_unit, axis=0) # sum over genomes
        # print ("5th of loss calculation: loss_by_unit.shape=", loss_by_unit.shape)
        #loss = tf.reduce_sum(loss_by_unit) # sum over profiles=units
        # ^^^ unused
        
        return score, loss_by_unit

    # return for each profile the best score at any position in the dataset
    def max_profile_scores(self, ds, otherP = None):
        U = self.units if otherP is None else otherP.shape[-1]
        scores = tf.ones([U], dtype=tf.float32) * -np.infty
        for batch, _ in ds:
            #assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                S, _, _ = self.call(X, otherP)                            # shape (ntiles, N, U)
                scores = tf.maximum(tf.reduce_max(S, axis=(0,1)), scores) # shape (U)
                                    
        return scores
    
    # return for each profile the loss contribution
    def profile_losses(self, ds, otherP = None):
        U = self.units if otherP is None else otherP.shape[-1]
        losses = tf.zeros([U, 0], dtype=tf.float32) # shape (U, 0)
        for batch, _ in ds:
            #assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                _, _, Z = self.call(X, otherP)
                _, losses_by_unit = self.loss(Z) # shape (U)
                #losses += losses_by_unit
                losses = tf.concat([losses, tf.expand_dims(losses_by_unit, -1)], axis=1) # shape (U, x) (where x is number of batches*batch_size)
                                    
        return losses
    
    def min_profile_losses(self, ds, otherP = None):
        lossesPerBatch = self.profile_losses(ds, otherP)   # (U, x)
        losses = tf.reduce_sum(lossesPerBatch, axis=-1)    # (U)
        return losses
    
    @tf.function()
    def train_step(self, X):
        with tf.GradientTape() as tape:
            S, R, Z = self.call(X)
            L, _ = self.loss(Z)
            # Mario's loss
            negscore = -L

        #grad = tape.gradient(L, self.P_logit)
        grad = tape.gradient(negscore, self.P_logit)
        self.opt.apply_gradients([(grad, self.P_logit)])
        
        return S, R, negscore#L

    # Mario's training
    def train_ds(self, ds, steps_per_epoch, epochs, verbose=True):
        self.opt = tf.keras.optimizers.Adam(learning_rate=1.) # large learning rate is much faster
        
        Lb = []
        Smin, Smax = float('inf'), float('-inf')
        Rmin, Rmax = float('inf'), float('-inf')
        
        for i in range(epochs):
            steps = 0
            for batch, _ in ds:
                for X in batch:
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
                    
            if verbose and (i%(25) == 0 or i==epochs-1):
                S, R, Z = self(X)
                score, _ = self.loss(Z)
                print(f"epoch {i:>5} score={score.numpy():.4f}" +
                      " max R: {:.3f}".format(tf.reduce_max(R).numpy()) +
                      " min R: {:.3f}".format((tf.reduce_min(R).numpy())))
                
            
    
    def train(self, genomes, dsHelper, #tiles_per_X, tile_size, batch_size, 
              steps_per_epoch, epochs, #prefetch=3,
              profile_plateau=5, profile_plateau_dev=1, match_score_factor=0.8,
              learning_rate=1.,
              n_best_profiles=None, verbose=True, verbose_freq=100):
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) # large learning rate is much faster

        def profileHistInit():
            return {
                'idx': np.ndarray([profile_plateau], dtype=np.int), 
                'score': np.ndarray([profile_plateau], dtype=np.int),
                'i': 0,
                'c': 0}

        tstart = time()
        i = 0
        profileHist = profileHistInit()
        run = True
        while run:
        #for i in range(epochs):
            steps = 0
            Lb = []
            Smin, Smax = float('inf'), float('-inf')
            Rmin, Rmax = float('inf'), float('-inf')
            #ds_train = dsg.getDataset(genomes, tiles_per_X, tile_size).repeat().batch(batch_size).prefetch(prefetch)
            #ds_eval = dsg.getDataset(genomes, tiles_per_X, tile_size, True).batch(batch_size).prefetch(prefetch)
            #ds_loss = dsg.getDataset(genomes, tiles_per_X, tile_size).batch(batch_size).prefetch(prefetch)
            #ds_cleanup = dsg.getDataset(genomes, tiles_per_X, tile_size, True).batch(batch_size).prefetch(prefetch)
            ds_train = dsHelper.getDataset(repeat = True)
            ds_eval = dsHelper.getDataset(withPosTracking = True)
            for batch, _ in ds_train:         # shape: (batchsize, ntiles, N, 6, tile_size, alphabet_size)
                #assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
                for X in batch:            # shape: (ntiles, N, 6, tile_size, alphabet_size)
                    assert len(X.shape) == 5, str(X.shape)
                    #print("[DEBUG] >>> performing train_step")
                    S, R, L = self.train_step(X)
                    #print("[DEBUG] >>> train_step performed")
                    Lb.append(L)
                    Rmax = max(Rmax, tf.reduce_max(R).numpy())
                    Rmin = min(Rmin, tf.reduce_min(R).numpy())
                    Smax = max(Smax, tf.reduce_max(S).numpy())
                    Smin = min(Smin, tf.reduce_min(S).numpy())
                    
                steps += 1
                if steps >= steps_per_epoch:
                    break
                    
            p, s = self.get_best_profile(ds_eval)
            #print("epoch", i, "best profile", p.numpy(), "with score", s.numpy())
            profileHist['idx'][profileHist['i']] = p
            profileHist['score'][profileHist['i']] = s.numpy()
            profileHist['i'] = 0 if profileHist['i']+1 == profile_plateau else profileHist['i']+1
            profileHist['c'] += 1
            if profileHist['c'] >= profile_plateau:
                if all(profileHist['idx'] == p):
                    sm = np.mean(profileHist['score'])
                    if all(np.logical_and(profileHist['score'] >= sm-profile_plateau_dev,
                                          profileHist['score'] <= sm+profile_plateau_dev)):
                        #ds_score = dsg.getDataset(genomes, tiles_per_X, tile_size).batch(batch_size).prefetch(prefetch)
                        #maxscores = self.max_profile_scores(dsHelper.getDataset())
                        #tau = match_score_factor * maxscores[p.numpy()]
                        print("epoch", i, "best profile", p.numpy(), "with score", s.numpy())
                        print("cleaning up profile", p.numpy())#, "with threshold", tau.numpy())
                        #self.profile_cleanup_bak(p, tau, dsHelper)
                        self.profile_cleanup(p, match_score_factor, dsHelper)
                        profileHist = profileHistInit()
                    
            self.history['loss'].append(np.mean(Lb))
            self.history['Rmax'].append(Rmax)
            self.history['Rmin'].append(Rmin)
            self.history['Smax'].append(Smax)
            self.history['Smin'].append(Smin)
                    
            if verbose and (i%(verbose_freq) == 0 or i==epochs-1):
                S, R, Z = self(X)
                L, _ = self.loss(Z)
                tnow = time()
                print("epoch", i, "best profile", p.numpy(), "with score", s.numpy())
                print(f"epoch {i:>5} loss={L.numpy():.4f}" +
                      " max R: {:.3f}".format(tf.reduce_max(R).numpy()) +
                      " min R: {:.3f}".format((tf.reduce_min(R).numpy())) +
                      " time: {:.2f}".format(tnow-tstart)) 

            i += 1
            if n_best_profiles is not None:
                run = (len(self.P_report) < n_best_profiles)
            else:
                run = (i < epochs)

    # old way without shift
    def profile_cleanup_bak(self, pIdx, threshold, dsHelper):
        losses = self.profile_losses(dsHelper.getDataset())[pIdx,:] # shape (x) where x is batch_size * number of batches
        # "remove" match sites from genomes
        matches, _ = self.get_profile_match_sites(dsHelper.getDataset(withPosTracking = True), 
                                                  threshold, pIdx) # site: (genomeID, contigID, pos, u)
        #print("DEBUG >>> matches:", matches)
        genomes = dsHelper.genomes
        for site in matches:
            #print("DEBUG >>> site:", site)
            g = site[0]
            c = site[1]
            a = site[2]
            b = a+(self.k*3)
            if a >= 0 and b <= len(genomes[g][c]):
                #print("DEBUG >>>  pre:", genomes[g][c][:a])
                #print("DEBUG >>>  new:", genomes[g][c][a:b].lower())
                #print("DEBUG >>> post:", genomes[g][c][b:])
                genomes[g][c] = genomes[g][c][:a]+genomes[g][c][a:b].lower()+genomes[g][c][b:] # mask match

        # report profile, get new seeds
        P = self.P_logit # shape: (k, alphabet_size, U)
        b = P[:,:,pIdx].numpy()
        self.P_report.append(b)
        self.P_report_thresold.append(threshold)
        #self.P_report_loss.append(tf.reduce_mean(losses).numpy())
        self.P_report_loss.append(tf.reduce_min(losses).numpy())
        self.P_logit.assign(self.seed_P_genome(genomes)) # completely new set of seeds
        
    def profile_cleanup(self, pIdx, match_score_factor, dsHelper):
        # get ks-mer, extract all k-mers, temporarily set k-mers as new profiles
        P = self.P_logit # shape: (k, alphabet_size, U)
        b = P[:,:,pIdx].numpy()
        Pk_logit = np.empty(shape=(self.k, self.alphabet_size, (2*self.s)+1), dtype=np.float32)
        for s in range(b.shape[0]-self.k+1):
            Pk_logit[:,:,s] = b[s:(s+self.k),:]
            
        Pk = tf.nn.softmax(Pk_logit, axis=1, name="Pk")
        #self.P_logit.assign(Pk)
        #Ubak = self.units
        #self.units = Pk.shape[2]
        
        # get best k-mer and report (unless it is the first or last k-mer when shit > 0)
        genomes = dsHelper.genomes
        #losses = self.min_profile_losses(dsHelper.getDataset(), otherP = Pk)   # (U)
        #bestIdx = tf.math.argmin(losses, axis=0).numpy()
        scores = self.max_profile_scores(dsHelper.getDataset(), otherP = Pk)   # (U)
        bestIdx = tf.math.argmax(scores, axis=0).numpy()
        if bestIdx not in [0, Pk.shape[2]-1] or self.s == 0:
            #threshold = match_score_factor * self.max_profile_scores(dsHelper.getDataset(), otherP = Pk)[bestIdx]
            threshold = match_score_factor * scores[bestIdx]
            minloss = tf.reduce_min(self.profile_losses(dsHelper.getDataset(), otherP = Pk)[bestIdx,:]).numpy()
            # "remove" match sites from genomes
            matches, _ = self.get_profile_match_sites(dsHelper.getDataset(withPosTracking = True), 
                                                      threshold, bestIdx, otherP = Pk) # site: (genomeID, contigID, pos, u)
            #print("DEBUG >>> matches:", matches)
            reportSites = []
            for site in matches:
                #print("DEBUG >>> site:", site)
                g = site[0]
                c = site[1]
                a = site[2]
                b = a+(self.k*3)
                
                # [DEBUG] report matched sites for each reported profile
                reportSites.append((dsg.sequence_translation(genomes[g][c][a:b].upper()), g, c, a, b))
                
                if a >= 0 and b <= len(genomes[g][c]):
                    #print("DEBUG >>>  pre:", genomes[g][c][:a])
                    #print("DEBUG >>>  new:", genomes[g][c][a:b].lower())
                    #print("DEBUG >>> post:", genomes[g][c][b:])
                    genomes[g][c] = genomes[g][c][:a]+genomes[g][c][a:b].lower()+genomes[g][c][b:] # mask match

            # report profile, get new seeds
            self.P_report.append(Pk_logit[:,:,bestIdx])
            self.P_report_whole.append(P[:,:,pIdx])
            self.P_report_thresold.append(threshold)
            #self.P_report_loss.append(tf.reduce_mean(losses).numpy())
            self.P_report_loss.append(minloss)
            self.P_report_masked_sites.append(reportSites)
            #self.P_report_kmer_losses.append(losses)
            self.P_report_kmer_scores.append(scores)

        # reset profiles
        #self.units = Ubak
        self.P_logit.assign(self.seed_P_genome(genomes)) # completely new set of seeds

    def seed_P_ds(self, ds):
        """
            Seed profiles P with profiles that represent units random positions in the input sequences.
            Positions are drawn uniformly from all positions in all sequences. 
            This is done with an old and neat trick online, so that the data has to be read only once.
        """
        rho = 2.0
        oneProfile_logit_like_Q = np.log(self.Q)
        P_logit_init = self._getRandomProfiles() # shape [k, alphabet_size, units]
        ks = self.k + (2*self.s) # trained profile width (k +- shift)
        m = 0 # number of positions seen so far
        for batch, _ in ds:
            #assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                X = X.numpy()                            # shape: (ntiles, N, 6, tile_size, alphabet_size)
                PP = X.reshape([-1, self.alphabet_size]) # shape: (ntiles*N*6*tile_size, alphabet_size)
                num_pos = PP.shape[0] - ks               # ntiles*N*6*tile_size - ks
                # PP[j,a] is 1 if the j-th character in an artificially concatenated sequence is char a
                # the length k patterns extend over tile ends, which could be improved later
                for j in range(num_pos):
                    i = -1 # i-th profile is to be replaced, unless i<0
                    if m < self.units:
                        i = m
                    elif np.random.choice(m) < self.units:
                        i = np.random.choice(self.units)
                    if i >= 0:
                        # replace i-th profile with a seed profile build from the pattern starting at position j
                        # Seed is the background distribution, except the observed k-mer at pos j is more likely
                        seed = rho * PP[j:j+ks,:] + oneProfile_logit_like_Q
                        P_logit_init[:,:,i] = seed
                    m += 1

        return P_logit_init
    
    def seed_P_triplets(self):
        self.units = self.alphabet_size ** 3
        P_logit = np.zeros((self.k+(2*self.s), self.alphabet_size, self.units), dtype=np.float32)
        p = 0
        for i in range(self.alphabet_size):
            for j in range(self.alphabet_size):
                for k in range(self.alphabet_size):
                    P_logit[0,:,p] = np.ones(self.alphabet_size, dtype=np.float32) * 1e-6
                    P_logit[1,:,p] = np.ones(self.alphabet_size, dtype=np.float32) * 1e-6
                    P_logit[2,:,p] = np.ones(self.alphabet_size, dtype=np.float32) * 1e-6
                    P_logit[0,i,p] = 1e6
                    P_logit[1,j,p] = 1e6
                    P_logit[2,k,p] = 1e6
                    p += 1
                    
        return P_logit
    
    def seed_P_genome(self, genomes):
        if True:
            flatg = []
            for seqs in genomes:
                flatg.extend(seqs) # should be all references and thus no considerable memory overhead

            lensum = sum([len(s) for s in flatg])
            weights = [len(s)/lensum for s in flatg]
            weights = tf.nn.softmax(weights).numpy()
            seqs = np.random.choice(len(flatg), self.units, replace=True, p=weights)
            ks = self.k + (2*self.s) # trained profile width (k +- shift)

            oneProfile_logit_like_Q = np.log(self.Q)
            P_logit_init = np.zeros((ks, self.alphabet_size, self.units), dtype=np.float32)
            rho = 2.0
            kdna = ks * 3
            for j in range(self.units):
                i = seqs[j] # seq index
                assert len(flatg[i]) > kdna, str(len(flatg[i]))
                pos = np.random.choice(len(flatg[i])-kdna, 1)[0]
                aa = dsg.sequence_translation(flatg[i][pos:pos+kdna])
                OH = dsg.oneHot(aa)
                assert OH.shape == (ks, self.alphabet_size), str(OH.shape)+" != "+str((ks, self.alphabet_size))
                seed = rho * OH + oneProfile_logit_like_Q
                P_logit_init[:,:,j] = seed

            return P_logit_init
        else:
            return self._getRandomProfiles()
                
    def setP_logit(self, P_logit_init):
        self.P_logit = tf.Variable(P_logit_init, trainable=True, name="P_logit") 