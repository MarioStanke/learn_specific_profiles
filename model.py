#!/usr/bin/env python

from operator import pos
import numpy as np
import os
#import pandas as pd
#import sequtils as su
#import seq
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
    def __init__(self, k, alphabet_size, units, Q, P_logit_init=None, **kwargs):
        super().__init__(**kwargs)
        
        self.Q = Q                         # shape: (alphabet_size)
        self.k = k                         # shape: ()
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
        
    def _getRandomProfiles(self):
        Q1 = tf.expand_dims(self.Q,0)
        Q2 = tf.expand_dims(Q1,-1)         # shape: (1, alphabet_size, 1)
        
        P_logit_like_Q = np.log(Q2.numpy())
        P_logit_init = P_logit_like_Q + np.random.normal(scale=4., size=[self.k, self.alphabet_size, self.units]).astype('float32')
        return P_logit_init                # shape: (k, alphabet_size, U)
        
    # return for each profile the best score at any position in the dataset
    def get_profile_match_sites_old(self, ds, threshold, aa_tile_size, genomes, L5score: bool = False):
        # dict of dicts of dict, for each genome, map each contig to a dict that collects profile indices and positions
        #   (given that the profile matches with score above threshold for that genome, contig and position)
        sites = {}
        for batch in ds:
            #assert len(batch) == 2, str(len(batch))+" -- use batch dataset with position tracking!"
            X = batch[0]        # (B, tilePerX, N, 6, tileSize, 21)
            posTrack = batch[1] # (B, tilePerX, N, 3)
            assert len(X.shape) == 6, str(X.shape)
            assert len(posTrack.shape) == 4, str(posTrack.shape)+" -- use batch dataset with position tracking!"
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

    def get_profile_match_sites(self, ds, score_threshold, pIdx = None):
        gamma = .2
        matches = None
        for batch in ds:
            #assert len(batch) == 2, str(len(batch))+" -- use batch dataset with position tracking!"
            X_b = batch[0]        # (B, tilePerX, N, 6, tileSize, 21)
            posTrack_b = batch[1] # (B, tilePerX, N, 4)
            assert len(X_b.shape) == 6, str(X_b.shape)
            assert len(posTrack_b.shape) == 4, str(posTrack_b.shape)+" -- use batch dataset with position tracking!"
            assert X_b.shape[0:3] == posTrack_b.shape[0:3], str(X_b.shape)+" != "+str(posTrack_b.shape)
            for b in range(X_b.shape[0]): # iterate samples in batch
                X = X_b[b]
                posTrack = posTrack_b[b]
                _, _, Z = self.call(X)                                             # (tilePerX, N, 6, T-k+1, U)
                if pIdx is not None:
                    Z = Z[:,:,:,:,pIdx:(pIdx+1)] # only single profile, but keep dimensions

                Z2 = tf.nn.softmax(gamma*Z, axis=0)
                Z3 = tf.math.multiply(Z, tf.square(Z2))
                
                Idx = indexTensor(Z.shape[0], Z.shape[1], Z.shape[2],
                                  Z.shape[3], Z.shape[4])                          # (tilePerX, N, 6, T-k+1, U, (f,r,u)) (f - frame, r - rel. pos., u - profile)

                # collapse genome and tile dimensions
                Z = tf.reshape(Z3, [-1, Z3.shape[-3], Z3.shape[-2], Z3.shape[-1]]) # (tilesPerX*N, 6, T-k+1, U)
                M = tf.greater_equal(Z, score_threshold)                           # (tilesPerX*N, 6, T-k+1, U), >>> consider match if score >= avg. profile score <<<
                T = tf.reshape(posTrack, [-1, 4])                                  # (tilesPerX*N, (genomeID, contigID, fwdStart, rcStart))
                Idx = tf.reshape(Idx, [-1, Idx.shape[-4], Idx.shape[-3], 
                                           Idx.shape[-2], Idx.shape[-1]])          # (tilesPerX*N, 6, T-k+1, U, (f,r,u))
                
                # reduce to genome tiles that have matches
                Mgentile = tf.reduce_any(M, axis=[1,2,3]) # (tilesPerX*N), of which `matches` are True
                Mgentile = tf.logical_and(Mgentile, tf.not_equal(T[:,0], -1)) # also set exhausted contigs to False
                T = tf.boolean_mask(T,Mgentile)           # (matches, (genomeID, contigID, fwdStart, rcStart))
                M = tf.boolean_mask(M,Mgentile)           # (matches, 6, T-k+1, U)
                Idx = tf.boolean_mask(Idx, Mgentile)      # (matches, 6, T-k+1, U, 3)

                # manual broadcast of T to the correct shape
                T = tf.repeat(tf.expand_dims(T, 1), [M.shape[-1]], axis=1) # (matches, U, (genomeID, contigID, fwdStart, rcStart))
                T = tf.repeat(tf.expand_dims(T, 1), [M.shape[-2]], axis=1) # (matches, T-k+1, U, (genomeID, contigID, fwdStart, rcStart))
                T = tf.repeat(tf.expand_dims(T, 1), [M.shape[-3]], axis=1) # (matches, 6, T-k+1, U, (genomeID, contigID, fwdStart, rcStart))
                
                # reduce to single match sites
                Idx = tf.boolean_mask(Idx, M)    # (sites, (f, r, u))
                T = tf.boolean_mask(T, M)        # (sites, (genomeID, contigID, fwdStart, rcStart))
                R = tf.concat((Idx, T), axis=1 ) # (sites, (f, r, u, genomeID, contigID, fwdStart, rcStart))

                # continue with vectorized operations, tf.map_fn to dsg.restoreGenomePosition is super slow even with parallel threads enabled
                fwdMask = tf.less(R[:,0], 3)            # (sites)
                rcMask = tf.greater_equal(R[:,0], 3)    # (sites)
                rFwd = tf.boolean_mask(R, fwdMask)      # (fwdSites, (f, r, u, genomeID, contigID, fwdStart, rcStart))
                rRC = tf.boolean_mask(R, rcMask)        # ( rcSites, (f, r, u, genomeID, contigID, fwdStart, rcStart))

                # fwd case: p *= 3; p += frame; p += tileStart
                posFwd = tf.multiply(rFwd[:,1], 3)             # (fwdSites), *3
                posFwd = tf.add(posFwd, rFwd[:,0])             # (fwdSites), add frame offset
                posFwd = tf.add(posFwd, rFwd[:,5])             # (fwdSites), add tile start
                sites = tf.concat([tf.expand_dims(rFwd[:,3], -1), 
                                   tf.expand_dims(rFwd[:,4], -1), 
                                   tf.expand_dims(posFwd, -1), 
                                   tf.expand_dims(rFwd[:,2], -1)], axis=1) # (fwdSites, (genomeID, contigID, pos, u))
                
                # rc case: p *= 3; p += frame-3; p = rcStart - p - (k*3) + 1
                posRC = tf.multiply(rRC[:,1], 3)               # (rcSites), *3
                posRC = tf.add(posRC, rRC[:,0])                # (rcSites), add frame offset
                posRC = tf.subtract(posRC, 3)                  # (rcSites), -3
                posRC = tf.subtract(rRC[:,6], posRC)           # (rcSites), rcStart - p
                posRC = tf.subtract(posRC, (self.k*3))         # (rcSites), -(k*3)
                posRC = tf.add(posRC, 1)                       # (rcSites), +1
                sitesRC = tf.concat([tf.expand_dims(rRC[:,3], -1), 
                                     tf.expand_dims(rRC[:,4], -1), 
                                     tf.expand_dims(posRC, -1), 
                                     tf.expand_dims(rRC[:,2], -1)], axis=1) # (fwdSites, (genomeID, contigID, pos, u))
                
                sites = tf.concat((sites, sitesRC), axis=0) # (sites, (genomeID, contigID, pos, u))
                if matches is None:
                    matches = sites
                else:
                    matches = tf.concat([matches, sites], axis=0)

        if matches is None:
            matches = tf.constant([])        
        
        return matches

    
    # return index of the profile that has the highest mean score of all scores in all tiles
    def get_best_profile(self, ds):
        Ms = []
        for batch in ds:
            #assert len(batch) == 2, str(len(batch))+" -- use batch dataset with position tracking!"
            X = batch[0]        # (B, tilePerX, N, 6, tileSize, 21)
            posTrack = batch[1] # (B, tilePerX, N, 3)
            assert len(X.shape) == 6, str(X.shape)
            assert len(posTrack.shape) == 4, str(posTrack.shape)+" -- use batch dataset with position tracking!"
            assert X.shape[0:3] == posTrack.shape[0:3], str(X.shape)+" != "+str(posTrack.shape)
            for b in range(X.shape[0]): # iterate samples in batch
                S, _, _ = self.call(X[b])               # S: (ntiles, N, U)
                gamma = .2
                S2 = tf.nn.softmax(gamma*S, axis=0)     #    (ntiles, N, U)
                S3 = tf.math.multiply(S, tf.square(S2)) #    (ntiles, N, U)
                M = tf.reduce_mean(S3, axis=[0,1])      #               (U), mean score of each profile

                W = tf.cast(posTrack[b,:,:,0] != -1, tf.float32) # tilePerX, N -> -1 if contig was exhausted -> False if exhausted -> 1 for valid contig, 0 else
                W1 = tf.multiply(tf.reduce_sum(W), tf.ones(shape = [self.units], dtype=tf.float32)) # weight for the means, shape (U)
                Ms.append(tf.multiply(M, W1).numpy()) # store weighted means

        B = tf.reduce_mean(Ms, axis=0) # get overall highest mean scores per profile
        return tf.argmax(B), tf.reduce_max(B) # index, mean score

    def getP(self):
        P = tf.nn.softmax(self.P_logit, axis=1, name="P")
        return P                           # shape: (k, alphabet_size, U)

    def getP_report(self):
        #P = tf.nn.softmax(
        P = tf.transpose(self.P_report, [1,2,0])#, axis=1, name="P")
        return P                           # shape: (k, alphabet_size, -1)
    
    def getR(self):
        P = self.getP()
        Q1 = tf.expand_dims(self.Q,0)
        Q2 = tf.expand_dims(Q1,-1)
        # Limit the odds-ratio, to prevent problem with log(0).
        # Very bad matches of profiles are irrelevant anyways.
        ratio = tf.maximum(P/Q2, 1e-6)
        R = tf.math.log(ratio)
        return R                           # shape: (k, alphabet_size, U)
    
    def getZ(self, X):
        R = self.getR()

        X1 = tf.expand_dims(X,-1) # 1 input channel   shape: (ntiles, N, 6, tile_size, alphabet_size, 1)
        R1 = tf.expand_dims(R,-2) # 1 input channel   shape: (k, alphabet_size, 1, U)

        # X1: (batch_shape (ntiles, N, 6), in_height (tile_size), in_width (alphabet_size),     in_channels (1))
        # R1:                             (filter_height (k),     filter_width (alphabet_size), in_channels (1), out_channels (U))
        # Z1: (batch_shape (ntiles, N, 6), tile_size-k+1, 1, U)
        Z1 = tf.nn.conv2d(X1, R1, strides=1,
                          padding='VALID', data_format="NHWC", name="Z")
        Z = tf.squeeze(Z1, 4) # remove input channel dimension   shape (ntiles, N, 6, tile_size-k+1, U)
        return Z, R
        
    def call(self, X):
        Z, R = self.getZ(X)

        S = tf.reduce_max(Z, axis=[2,3])   # shape (ntiles, N, U)
        return S, R, Z

    # custom loss
    def loss(self, S):
        # penalize multiple similarly good near-best matches in the same genome
        gamma = .2 # a small value means a more inclusive meaning of near-best
        S1 = tf.nn.softmax(gamma*S, axis=0) # shape (ntiles, N, U)
        S2 = tf.reduce_max(S1, axis=0) # the closer to 1, the clearer is the champion match a winner   shape (N, U)
        S3 = tf.reduce_max(S, axis=0) # ranges over tiles, or soft max like in L1                      shape (N, U)
        S4 = tf.math.multiply(S3, tf.square(S2)) # effectively the best score per genome is divided by the number of matches
        loss_by_unit = tf.reduce_sum(-S4, axis=0) / self.units # sum over genomes                      shape (U)
        L = tf.reduce_sum(loss_by_unit) # sum over profiles=units
        
        return L, loss_by_unit             # shape: (), (U)

    # return for each profile the best score at any position in the dataset
    def max_profile_scores(self, ds):
        scores = tf.ones([self.units], dtype=tf.float32) * -np.infty
        for batch, _ in ds:
            #assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                S, _, _ = self.call(X)
                scores = tf.maximum(tf.reduce_max(S, axis=(0,1)), scores)
                                    
        return scores
    
    # return for each profile the loss contribution
    def min_profile_losses(self, ds):
        losses = tf.zeros([self.units], dtype=tf.float32)
        for batch, _ in ds:
            #assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                S, _, _ = self.call(X)
                _, losses_by_unit = self.loss(S)
                losses += losses_by_unit
                                    
        return losses
    
    @tf.function()
    def train_step(self, X):
        with tf.GradientTape() as tape:
            S, R, _ = self.call(X)
            L, _ = self.loss(S)

        grad = tape.gradient(L, self.P_logit)
        self.opt.apply_gradients([(grad, self.P_logit)])
        
        return S, R, L

    def train(self, genomes, tiles_per_X, tile_size, 
              batch_size, steps_per_epoch, epochs, prefetch=3,
              profile_plateau=5, profile_plateau_dev=1, n_best_profiles=None, verbose=True, verbose_freq=100):
        self.opt = tf.keras.optimizers.Adam(learning_rate=1.) # large learning rate is much faster

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
            ds_train = dsg.getDataset(genomes, tiles_per_X, tile_size).repeat().batch(batch_size).prefetch(prefetch)
            ds_eval = dsg.getDataset(genomes, tiles_per_X, tile_size, True).batch(batch_size).prefetch(prefetch)
            ds_seed = dsg.getDataset(genomes, tiles_per_X, tile_size).batch(batch_size).prefetch(prefetch)
            ds_cleanup = dsg.getDataset(genomes, tiles_per_X, tile_size, True).batch(batch_size).prefetch(prefetch)
            #ds_dbg = dsg.getDataset(genomes, tiles_per_X, tile_size, True).batch(batch_size).prefetch(prefetch)
            #ds_dbg2 = dsg.getDataset(genomes, tiles_per_X, tile_size, True).batch(batch_size).prefetch(prefetch)
            for batch, _ in ds_train:         # shape: (batchsize, ntiles, N, 6, tile_size, alphabet_size)
                #assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
                for X in batch:            # shape: (ntiles, N, 6, tile_size, alphabet_size)
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
                    
            p, s = self.get_best_profile(ds_eval)
            print("epoch", i, "best profile", p.numpy(), "with score", s.numpy())
            profileHist['idx'][profileHist['i']] = p
            profileHist['score'][profileHist['i']] = s
            profileHist['i'] = 0 if profileHist['i']+1 == profile_plateau else profileHist['i']+1
            profileHist['c'] += 1
            if profileHist['c'] >= profile_plateau:
                if all(profileHist['idx'] == p):
                    sm = np.mean(profileHist['score'])
                    if all(np.logical_and(profileHist['score'] >= sm-profile_plateau_dev,
                                          profileHist['score'] <= sm+profile_plateau_dev)):
                        tau = 0.8*self.k # arbitrary threshold
                        print("cleaning up profile", p.numpy(), "with threshold", tau)
                        #print("[DEBUG]  >>> matches of "+str(p.numpy())+" with threshold "+str(tau)+":", self.get_profile_match_sites(ds_dbg, tau, p))
                        #print("[DEBUG2] >>> matches of "+str(p.numpy())+" with threshold "+str(tau)+":", self.get_profile_match_sites(ds_dbg2, tau, p))
                        self.profile_cleanup(p, tau, genomes, ds_seed, ds_cleanup)
                        profileHist = profileHistInit()
                    
            self.history['loss'].append(np.mean(Lb))
            self.history['Rmax'].append(Rmax)
            self.history['Rmin'].append(Rmin)
            self.history['Smax'].append(Smax)
            self.history['Smin'].append(Smin)
                    
            if verbose and (i%(verbose_freq) == 0 or i==epochs-1):
                S, R, _ = self(X)
                L, _ = self.loss(S)
                tnow = time()
                print(f"epoch {i:>5} loss={L.numpy():.4f}" +
                      " max R: {:.3f}".format(tf.reduce_max(R).numpy()) +
                      " min R: {:.3f}".format((tf.reduce_min(R).numpy())) +
                      " time: {:.2f}".format(tnow-tstart)) 

            i += 1
            if n_best_profiles is not None:
                run = (len(self.P_report) < n_best_profiles)
            else:
                run = (i < epochs)

    def profile_cleanup(self, pIdx, threshold, genomes, ds_seed, ds_clean):
        # "remove" match sites from genomes
        matches = self.get_profile_match_sites(ds_clean, threshold, pIdx) # site: (genomeID, contigID, pos, u)
        #print("DEBUG >>> matches:", matches)
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
        P = self.getP() # shape: (k, alphabet_size, U)
        b = P[:,:,pIdx].numpy()
        #tau = self.k * 0.8 # arbitrary threshold, for now require 80% "similarity"
        #sim = []
        ## find similar profiles
        #for p in range(self.units):
        #    if smith_waterman(P[:,:,p].numpy(), b) >= tau:
        #        sim.append(p)

        #Prand = self._getRandomProfiles()
        #P_logit_new = self.P_logit.numpy()
        #P_logit_new[:,:,sim] = Prand[:,:,sim] # replace similar profiles with new ones
        #self.P_logit.assign(P_logit_new)
        self.P_report.append(b)
        self.P_logit.assign(self.seed_P_ds(ds_seed)) # completely new set of seeds


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
        for batch, _ in ds:
            #assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                X = X.numpy()                            # shape: (ntiles, N, 6, tile_size, alphabet_size)
                PP = X.reshape([-1, self.alphabet_size]) # shape: (ntiles*N*6*tile_size, alphabet_size)
                num_pos = PP.shape[0] - self.k           # ntiles*N*6*tile_size - k
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
                        seed = rho * PP[j:j+self.k,:] + oneProfile_logit_like_Q
                        P_logit_init[:,:,i] = seed
                    m += 1

        return P_logit_init
    
    def seed_P_triplets(self):
        self.units = self.alphabet_size ** 3
        P_logit = np.zeros((self.k, self.alphabet_size, self.units), dtype=np.float32)
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
                
    def setP_logit(self, P_logit_init):
        self.P_logit = tf.Variable(P_logit_init, trainable=True, name="P_logit") 




# tried using more @tf.function to get more efficient execution, did not work (-> a few seconds slower)

# class SpecificProfile(tf.keras.Model):
#     def __init__(self, k, alphabet_size, units, Q, P_logit_init=None, gamma=0.2, **kwargs):
#         super().__init__(**kwargs)
#         # P_logit_init = tf.random.normal([k, alphabet_size, units], stddev=.5, dtype=tf.float32, seed=1)
        
#         self.gamma = tf.constant(gamma, dtype = tf.float32) # a small value means a more inclusive meaning of near-best (in loss)
#         self.Q = tf.constant(Q, shape = (alphabet_size,), dtype = tf.float32)
#         self.k = tf.constant(k, dtype= tf.int32)
#         self.alphabet_size = tf.constant(alphabet_size, dtype = tf.int32)
#         self.units = tf.constant(units, dtype = tf.int32)
#         self.unitsf = tf.constant(units, dtype = tf.float32) # need float for later calculations
#         self.opt = tf.keras.optimizers.Adam(learning_rate=1.) # large learning rate is much faster
#         self.history = {'loss': [],
#                         'Rmax': [],
#                         'Rmin': [],
#                         'Smax': [],
#                         'Smin': []}
        
#         if P_logit_init is None:
#             P_logit_init = self._getRandomProfiles()
                
#         self.setP_logit(P_logit_init)

#     def _getRandomProfiles(self):
#         Q1 = tf.expand_dims(self.Q,0)
#         Q2 = tf.expand_dims(Q1,-1) # shape: (1, alphabet_size, 1) 
        
#         P_logit_like_Q = np.log(Q2.numpy())
#         P_logit_init = P_logit_like_Q + np.random.normal(scale=4., size=[self.k, self.alphabet_size, self.units]).astype('float32')
#         return P_logit_init        # shape: (k, alphabet_size, U)
        
#     # return for each profile the best score at any position in the dataset
#     def get_profile_match_sites(self, ds, threshold, aa_tile_size, genomes, L5score: bool = False):
#         # dict of dicts of dict, for each genome, map each contig to a dict that collects profile indices and positions
#         #   (given that the profile matches with score above threshold for that genome, contig and position)
#         sites = {}
#         for batch in ds:
#             assert len(batch) == 2, str(len(batch))+" -- use batch dataset with position tracking!"
#             X = batch[0]        # (B, tilePerX, N, 6, tileSize, 21)
#             posTrack = batch[1] # (B, tilePerX, N, 3)
#             assert len(X.shape) == 6, str(X.shape)
#             assert len(posTrack.shape) == 4, str(posTrack.shape)
#             assert X.shape[0:3] == posTrack.shape[0:3], str(X.shape)+" != "+str(posTrack.shape)
#             for b in range(X.shape[0]): # iterate samples in batch
#                 # get positions of best profile matches
#                 _, _, Z = self.call(X[b])              # (tilePerX, N, 6, T-k+1, U)
#                 if L5score:
#                     gamma = .2
#                     Z2 = tf.nn.softmax(gamma*Z, axis=0)
#                     Z3 = tf.math.multiply(Z, tf.square(Z2))
#                     Z = Z3
                    
#                 for g in range(X.shape[2]):
#                     if tf.reduce_max(Z[:,g,:,:,:]) >= threshold:
#                         p = tf.argmax( tf.reduce_max(Z[:,g,:,:,:], axis=[0,1,2]) ).numpy() # index of the profile with best hit
#                         t = tf.argmax( tf.reduce_max(Z[:,g,:,:,p], axis=[1,2]) ).numpy()   # index of the tile with best hit
#                         f = tf.argmax( tf.reduce_max(Z[t,g,:,:,p], axis=[1]) ).numpy()     # index of the frame with best hit
#                         r = tf.argmax( Z[t,g,f,:,p] ).numpy()                              # index of the rel. position with best hit

#                         c = posTrack[b,t,g,0].numpy()
#                         if c < 0:
#                             continue # either bug or profile matches good on padding
                            
#                         assert c < len(genomes[g]), str(c)+" >= "+str(len(genomes[g]))+" for genome "+str(g)
                            
#                         cStart = posTrack[b,t,g,1].numpy()
#                         tilelen = posTrack[b,t,g,2].numpy()
                        
#                         pos = r*3 # to dna coord
#                         if f < 3:
#                             pos += f # add frame shift
#                             pos = cStart + pos
#                         else:
#                             # add appropriate frame shift
#                             seqlen = len(genomes[g][c])
#                             if seqlen%3 == tilelen%3:
#                                 pos += f-3
#                             else:
#                                 pos += dsg.rcFrameOffsets(seqlen)[f-3]
                                
#                             pos = tilelen - pos - (self.k*3)

#                         if g not in sites:
#                             sites[g] = {}
#                         if c not in sites[g]:
#                             sites[g][c] = {'profile': [], 'pos': [], 'score': [], 'frame': []}

#                         sites[g][c]['profile'].append(p)
#                         sites[g][c]['pos'].append(pos)
#                         sites[g][c]['score'].append(Z[t,g,f,r,p].numpy())
#                         sites[g][c]['frame'].append(f)
                        
#         return sites
    
#     # return index of the profile that has the highest mean score of all scores in all tiles
#     def get_best_profile(self, ds):
#         N = np.zeros(self.units, dtype=np.float32)
#         S = np.zeros(self.units, dtype=np.float32)
        
#         for batch in ds:
#             assert len(batch) == 2, str(len(batch))+" -- use batch dataset with position tracking!"
#             X = batch[0]        # (B, tilePerX, N, 6, tileSize, 21)
#             posTrack = batch[1] # (B, tilePerX, N, 3)
#             assert len(X.shape) == 6, str(X.shape)
#             assert len(posTrack.shape) == 4, str(posTrack.shape)
#             assert X.shape[0:3] == posTrack.shape[0:3], str(X.shape)+" != "+str(posTrack.shape)
#             for b in range(X.shape[0]): # iterate samples in batch
#                 # get positions of best profile matches
#                 _, _, Z = self.call(X[b])              # Z: tilePerX, N, 6, T-k+1, U), X: (batchsize, tilePerX, N, 6, T, alphSize) 
#                 gamma = .2
#                 Z2 = tf.nn.softmax(gamma*Z, axis=0)
#                 Z3 = tf.math.multiply(Z, tf.square(Z2))
#                 Z = Z3

#                 for u in range(Z.shape[-1]):
#                     for t in range(Z.shape[0]):
#                         s = tf.reduce_max(Z[t,:,:,:,u])
#                         n = tf.argmax( tf.reduce_max(Z[t,:,:,:,u], axis=[1,2]) ).numpy() # index of genome with highest score
#                         if posTrack[b,t,n,0] >= 0: # otherwise contig was exhausted
#                             N[u] += 1
#                             S[u] += s
                            
#         M = np.array([S[i]/N[i] if N[i] > 0 else -np.inf for i in range(len(N))])
#         return np.argmax(M), np.max(M) # index, mean score

#     def getP(self):
#         P = tf.nn.softmax(self.P_logit, axis=1, name="P")
#         return P
    
#     def getR(self):
#         P = self.getP()
#         Q1 = tf.expand_dims(self.Q,0)
#         Q2 = tf.expand_dims(Q1,-1)
#         # Limit the odds-ratio, to prevent problem with log(0).
#         # Very bad matches of profiles are irrelevant anyways.
#         ratio = tf.maximum(P/Q2, 1e-6)
#         R = tf.math.log(ratio)
#         return R
    
#     def getZ(self, X):
#         R = self.getR()

#         X1 = tf.expand_dims(X,-1) # 1 input channel
#         R1 = tf.expand_dims(R,-2) # 1 input channel
#         Z1 = tf.nn.conv2d(X1, R1, strides=1,
#                           padding='VALID', data_format="NHWC", name="Z")
#         Z = tf.squeeze(Z1, 4) # remove input channel dimension
#         return Z, R
    
#     @tf.function()
#     def call(self, X):
#         P = tf.nn.softmax(self.P_logit, axis=1, name="P") # shape: (k, alphabet_size, U)
#         Q1 = tf.expand_dims(self.Q, 0)
#         Q2 = tf.expand_dims(Q1, -1)                       # shape: (1, alphabet_size, 1)
#         # Limit the odds-ratio, to prevent problem with log(0).
#         # Very bad matches of profiles are irrelevant anyways.
#         ratio = tf.maximum(P/Q2, 1e-6)
#         R = tf.math.log(ratio)                            # shape: (k, alphabet_size, U)
#         X1 = tf.expand_dims(X,-1) # 1 input channel         shape: (ntiles, N, 6, tile_size, alphabet_size, 1)
#         R1 = tf.expand_dims(R,-2) # 1 input channel         shape: (k, alphabet_size, 1, U)

#         # X1: (batch_shape (ntiles, N, 6), in_height (tile_size), in_width (alphabet_size),     in_channels (1))
#         # R1:                             (filter_height (k),     filter_width (alphabet_size), in_channels (1), out_channels (U))
#         # Z1: (batch_shape (ntiles, N, 6), tile_size-k+1, 1, U)
#         Z1 = tf.nn.conv2d(X1, R1, strides=1,
#                           padding='VALID', data_format="NHWC", name="Z")
#         Z = tf.squeeze(Z1, 4) # remove input channel dimension, shape (ntiles, N, 6, tile_size-k+1, U)
#         S = tf.reduce_max(Z, axis=[2,3])                      # shape (ntiles, N, U)
#         return S, R, Z

#     # custom loss
#     @tf.function()
#     def loss(self, S):
#         # penalize multiple similarly good near-best matches in the same genome
#         S1 = tf.nn.softmax(tf.math.multiply(self.gamma, S), axis=0)
#         S2 = tf.reduce_max(S1, axis=0) # the closer to 1, the clearer is the champion match a winner
#         S3 = tf.reduce_max(S,  axis=0) # ranges over tiles, or soft max like in L1
#         S4 = tf.math.multiply(S3, tf.square(S2)) # effectively the best score per genome is divided by the number of matches
#         loss_by_unit = tf.math.multiply(tf.math.reciprocal(self.unitsf), tf.reduce_sum(-S4, axis=0))  # sum over genomes, divide by number of units
#         L5 = tf.reduce_sum(loss_by_unit) # sum over profiles=units
        
#         return L5, loss_by_unit

#     # return for each profile the best score at any position in the dataset
#     def max_profile_scores(self, ds):
#         scores = np.ones([self.units], dtype=np.float32) * -np.infty
#         for batch in ds:
#             assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
#             for X in batch:
#                 assert len(X.shape) == 5, str(X.shape)
#                 S, _, _ = self.call(X)
#                 scores = np.maximum(np.max(S, axis=(0,1)), scores)
                                    
#         return scores
    
#     # return for each profile the loss contribution
#     def min_profile_losses(self, ds):
#         losses = np.zeros([self.units], dtype=np.float32)
#         for batch in ds:
#             assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
#             for X in batch:
#                 assert len(X.shape) == 5, str(X.shape)
#                 S, _, _ = self.call(X)
#                 _, losses_by_unit, _ = self.loss(S)
#                 losses += losses_by_unit
                                    
#         return losses
    
#     @tf.function()
#     def train_step(self, X):
#         with tf.GradientTape() as tape:
#             S, R, _ = self.call(X)
#             L, _ = self.loss(S)

#         grad = tape.gradient(L, self.P_logit)
#         self.opt.apply_gradients([(grad, self.P_logit)])
        
#         return S, R, L

#     def train(self, ds_train, ds_eval, steps_per_epoch, epochs, verbose=True, verbose_freq=100):
#         tstart = time()
#         for i in range(epochs):
#             steps = 0
#             Lb = []
#             Smin, Smax = float('inf'), float('-inf')
#             Rmin, Rmax = float('inf'), float('-inf')
#             for batch in ds_train:
#                 assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
#                 for X in batch:
#                     assert len(X.shape) == 5, str(X.shape)
#                     S, R, L = self.train_step(X)
#                     Lb.append(L)
#                     Rmax = max(Rmax, tf.reduce_max(R).numpy())
#                     Rmin = min(Rmin, tf.reduce_min(R).numpy())
#                     Smax = max(Smax, tf.reduce_max(S).numpy())
#                     Smin = min(Smin, tf.reduce_min(S).numpy())
                    
#                 steps += 1
#                 if steps >= steps_per_epoch:
#                     break
                    
#             #p, s = self.get_best_profile(ds_eval)
#             #print("epoch", i, "best profile", p, "with score", s)
                    
#             self.history['loss'].append(np.mean(Lb))
#             self.history['Rmax'].append(Rmax)
#             self.history['Rmin'].append(Rmin)
#             self.history['Smax'].append(Smax)
#             self.history['Smin'].append(Smin)
                    
#             if verbose and (i%(verbose_freq) == 0 or i==epochs-1):
#                 S, R, _ = self.call(X)
#                 L, _ = self.loss(S)
#                 tnow = time()
#                 print(f"epoch {i:>5} loss={L.numpy():.4f}" +
#                       " max R: {:.3f}".format(tf.reduce_max(R).numpy()) +
#                       " min R: {:.3f}".format((tf.reduce_min(R).numpy())) +
#                       " time: {:.2f}".format(tnow-tstart)) 

#     def seed_P_ds(self, ds):
#         """
#             Seed profiles P with profiles that represent units random positions in the input sequences.
#             Positions are drawn uniformly from all positions in all sequences. 
#             This is done with an old and neat trick online, so that the data has to be read only once.

#             Tried transforming this to a @tf.function and using all tensors, turned out super slow compared to this!
#         """
#         rho = 2.0
#         oneProfile_logit_like_Q = np.log(self.Q.numpy())
#         P_logit_init = self._getRandomProfiles()#.numpy() # shape [k, alphabet_size, units]
#         m = 0 # number of positions seen so far
#         alphabet_size = self.alphabet_size.numpy() # avoid repetitive tensor-to-numpy conversions
#         k = self.k.numpy()
#         units = self.units.numpy()
#         for batch in ds:
#             assert len(batch.shape) == 6, str(batch.shape)+" -- use batch dataset without position tracking!"
#             for X in batch:
#                 assert len(X.shape) == 5, str(X.shape)
#                 X = X.numpy()
#                 PP = X.reshape([-1, alphabet_size]) # (tilesPerX, N, 6, T, alphSize) -> (tilesPerX*N*6*T, alphSize)
#                 J = PP.shape[0]                          # tilesPerX*N*6*T
#                 # PP[j,a] is 1 if the j-th character in an artificially concatenated sequence is char a
#                 # the length k patterns extend over tile ends, which could be improved later
#                 num_pos = J - k
#                 for j in range(num_pos):
#                     i = -1 # i-th profile is to be replaced, unless i<0
#                     if m < units:
#                         i = m
#                     elif np.random.choice(m) < units:
#                         i = np.random.choice(units)
#                     if i >= 0:
#                         # replace i-th profile with a seed profile build from the pattern starting at position j
#                         # Seed is the background distribution, except the observed k-mer at pos j is more likely
#                         seed = rho * PP[j:j+k,:] + oneProfile_logit_like_Q
#                         P_logit_init[:,:,i] = seed
#                         # print (f"replacing {i:>3}-th pattern with the one from pos {j:>6}:\n", seed)
#                     m += 1

#         return P_logit_init
    
#     def seed_P_triplets(self):
#         self.units = self.alphabet_size ** 3
#         P_logit = np.zeros((self.k, self.alphabet_size, self.units), dtype=np.float32)
#         p = 0
#         for i in range(self.alphabet_size):
#             for j in range(self.alphabet_size):
#                 for k in range(self.alphabet_size):
#                     P_logit[0,:,p] = np.ones(self.alphabet_size, dtype=np.float32) * 1e-6
#                     P_logit[1,:,p] = np.ones(self.alphabet_size, dtype=np.float32) * 1e-6
#                     P_logit[2,:,p] = np.ones(self.alphabet_size, dtype=np.float32) * 1e-6
#                     P_logit[0,i,p] = 1e6
#                     P_logit[1,j,p] = 1e6
#                     P_logit[2,k,p] = 1e6
#                     p += 1
                    
#         return P_logit

#     def setP_logit(self, P_logit_init):
#         self.P_logit = tf.Variable(P_logit_init, trainable=True, name="P_logit") 