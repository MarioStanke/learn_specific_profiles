#!/usr/bin/env python

import numpy as np
import dataset as dsg
import tensorflow as tf
from time import time



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

def indexTensor(T, N, F, P, U):
    """ Create index tensor for a data batch of shape (T, N, F, P, U). 
        Returns tensor of shape (T, N, F, P, U, 3) where each element of the second-last dimension is a list with 
        indices [f,p,u] (frame, rel.pos. profile) """
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
    def __init__(self, k, alphabet_size, units, Q, 
                 P_logit_init=None, alpha=1e-6, gamma=0.2, l2 = 1, 
                 shift=0, loss = "softmax", 
                 track_profiles = None,
                 rand_seed=None, **kwargs):
        """
        Set up model and most metaparamters
            Parameters:
                k (int): width of trained profiles
                alphabet_size (int): number of possible symbols in profiles
                units (int): number of profiles to use during training
                Q (list(float)): background distribution
                P_logit_init (np.ndarray(float)): optional numpy array giving initial set of profiles 
                                                  (shape (k[+2*shift], alphabet_size, units))
                alpha (float): learning rate
                gamma (float): sets effect of softmax in loss
                l2 (float): scale the L2 regularization term (currently only in experiment loss)
                shift (int): learn or evaluate shifts of learned profiles
                loss (str): either of 'softmax' or 'score' ('score' turns off softmax in loss function, so loss becomes
                                                              simply the negative score)
                track_profiles (list of int): indices of profiles to track, i.e. after each training step store the
                                                intermediate profile and the max scores
                rand_seed (int): optional set a seed for tensorflow's rng
        """
        super().__init__(**kwargs)
        
        assert loss in ['softmax', 'score', 'experiment'], \
            "[ERROR] >>> loss must be either 'softmax', 'score' or 'experiment'"
        
        if rand_seed is not None:
            print("[DEBUG] >>> setting tf global seed to", rand_seed)
            tf.random.set_seed(rand_seed)
            
        self.nprng = np.random.default_rng(rand_seed) # if rand_seed is None, unpredictable entropy is pulled from OS
        #np.random.seed(rand_seed)
        
        self.Q = Q                         # shape: (alphabet_size)
        self.alpha = alpha
        self.gamma = gamma # a small value means a more inclusive meaning of near-best 
                           #   ([3, 4, 1] -> [0.25, 0.7, 0.05] // [.3, .4, .1] -> [0.34, 0.38, 0.28])
        self.l2 = l2
        #perfect_match_score = np.log(1/np.mean(Q))  # set epsilon to a value that mismatch score is roughly -1*perfect match score
        #self.epsilon = np.exp(-perfect_match_score) # original value was 1e-6, which led to mismatch score of -13.8 while perfect match score is only 3
        self.epsilon = 1e-6
        self.k = k
        self.s = shift
        self.alphabet_size = alphabet_size
        self.units = units
        self.softmaxLoss = (loss == 'softmax')
        self.experimentLoss = (loss == 'experiment')
        print("[DEBUG] >>> using softmaxLoss:", self.softmaxLoss, "// using experimentLoss:", self.experimentLoss)
        self.track_profiles = track_profiles
        self.rand_seed = rand_seed
        
        self.history = {'loss': [],
                        'Rmax': [],
                        'Rmin': [],
                        'Smax': [],
                        'Smin': [],
                        'learning rate': []}
        
        if P_logit_init is None:
            P_logit_init = self._getRandomProfiles()
            self.P_logit_init = None
        else:
            print("[DEBUG] >>> Using P_logit_init from arguments instead of random")
            self.P_logit_init = P_logit_init
                
        self.setP_logit(P_logit_init)      # shape: (k, alphabet_size, U)
        self.P_report = []
        self.P_report_idx = []
        self.P_report_discarded = [] # for deleted edge cases
        self.P_report_thresold = []
        self.P_report_loss = []
        self.P_report_masked_sites = []
        self.P_report_nlinks = []
        
        self.P_report_whole = []
        self.P_report_whole_score = []
        self.P_report_whole_loss = []
                
        self.P_report_kmer_scores = []
        
        self.P_report_plosshist = None
        self.P_report_bestlosshist = []
        self.P_report_bestlosshistIdx = []
        
        # [DEBUG] store tracked profiles here
        self.tracking = {
            'epoch': [],
            'P': [], # list of np.arrays of all tracked profiles (k x 21 x U') where U'=len(track_profiles)
            'max_score': [], # list of np.arrays of respective max scores (U')
            'masking': [] # lookup sites in self.P_report_masked_sites
        }
        
        # initial state
        if self.track_profiles is not None and len(self.track_profiles) > 0:
            Pt = tf.gather(self.getP(), self.track_profiles, axis=2)
            self.tracking['epoch'].append(0)
            self.tracking['P'].append(Pt)
            self.tracking['max_score'].append(tf.constant(np.zeros(Pt.shape[2]), dtype=tf.float32))

    

    def setP_logit(self, P_logit_init):
        self.P_logit = tf.Variable(P_logit_init, trainable=True, name="P_logit") 
        
        
        
    def _getRandomProfiles(self):
        Q1 = tf.expand_dims(self.Q,0)
        Q2 = tf.expand_dims(Q1,-1)    # shape: (1, alphabet_size, 1)
        
        P_logit_like_Q = np.log(Q2.numpy())
        P_logit_init = P_logit_like_Q + self.nprng.normal(scale=4., size=[self.k+(2*self.s), self.alphabet_size, 
                                                                          self.units]).astype('float32')
        return P_logit_init           # shape: (self.k+(2*self.s), alphabet_size, U)
        
        
        
    def get_profile_match_sites(self, ds, score_threshold, pIdx = None, calculateLinks = False, otherP = None):
        """
        Get sites in the dataset where either all or a specific profile match according to a score threshold
            Parameters:
                ds: tf dataset
                score_threshold (float or tensor): matching sites need to achieve at least this score
                pIdx (int): optional index of a single profile, if given only matching sites of that profile are 
                              reported
                calculateLinks (int): flag, if True the number of possible links is computed and returned for each 
                                        profile
                otherP (profile tensor): optional profile tensor if not the model's profiles should be used
        """
        #print("[DEBUG] >>> score_threshold:", score_threshold)
        #print("[DEBUG] >>> pIdx:", pIdx)
        #if otherP is not None:
        #    print("[DEBUG] >>> otherP.shape:", otherP.shape)
        
        score_threshold = tf.convert_to_tensor(score_threshold)
        if otherP is not None:
            assert score_threshold.shape in [(), (otherP.shape[-1])], f"{score_threshold}, {score_threshold.shape}"
        else:
            assert score_threshold.shape in [(),(self.P_logit.shape[-1])], f"{score_threshold}, {score_threshold.shape}"
            
        matches = None
        scores = None
        for batch in ds:
            X_b = batch[0]        # (B, tilePerX, N, 6, tileSize, 21)
            posTrack_b = batch[1] # (B, tilePerX, N, 6, 4)
            assert len(X_b.shape) == 6, str(X_b.shape)
            assert posTrack_b.shape != (1, 0), str(posTrack.shape)+" -- use batch dataset with position tracking!"
            assert X_b.shape[0:4] == posTrack_b.shape[0:4], str(X_b.shape)+" != "+str(posTrack_b.shape)
            pTdim = posTrack_b.shape[-1]
            for b in range(X_b.shape[0]): # iterate samples in batch
                X = X_b[b]
                posTrack = posTrack_b[b]                   # (tilePerX, N, 6, (genomeID, contigID, startPos, aa_seqlen))
                _, _, Z = self.call(X, otherP) # automatically uses self.getP if otherP is None
                                                           # (tilePerX, N, 6, T-k+1, U)
                if pIdx is not None:
                    Z = Z[:,:,:,:,pIdx:(pIdx+1)] # only single profile, but keep dimensions
                
                Idx = indexTensor(Z.shape[0], Z.shape[1], Z.shape[2],
                                  Z.shape[3], Z.shape[4])  # (tilePerX, N, 6, T-k+1, U, (f,r,u)) 
                                                           #                     (f - frame, r - rel. pos., u - profile)

                # collapse genome and tile dimensions
                Z = tf.reshape(Z, [-1, Z.shape[-3], Z.shape[-2], Z.shape[-1]])  # (tilesPerX*N, 6, T-k+1, U)
                #print("[DEBUG] >>> Z.shape:", Z.shape)
                #print("[DEBUG] >>> score_threshold.shape:", score_threshold.shape)
                if score_threshold.shape != ():
                    score_threshold = tf.broadcast_to(score_threshold, Z.shape) # (tilesPerX*N, 6, T-k+1, U)
                #print("[DEBUG] >>> score_threshold.shape:", score_threshold.shape)
                M = tf.greater_equal(Z, score_threshold)                        # (tilesPerX*N, 6, T-k+1, U), 
                                                                          # >>> consider match if score >= threshold <<<
                #print("[DEBUG] >>> M.shape:", M.shape)
                #print("[DEBUG] >>> M reduce any:", tf.reduce_any(M))
                T = tf.reshape(posTrack, [-1, 6, pTdim])                        # (tilesPerX*N, 6, (genomeID, contigID, 
                                                                                #                  startPos, aa_seqlen))
                Idx = tf.reshape(Idx, [-1, Idx.shape[-4], Idx.shape[-3], 
                                           Idx.shape[-2], Idx.shape[-1]])       # (tilesPerX*N, 6, T-k+1, U, (f,r,u))
                
                # reduce to genome tiles that have matches
                Mgentile = tf.reduce_any(M, axis=[2,3])     # (tilesPerX*N, 6)
                Mgentile = tf.logical_and(Mgentile, tf.not_equal(T[:,:,0], -1)) # also set exhausted contigs to False
                Mgentile = tf.reduce_any(Mgentile, axis=1)  # (tilesPerX*N), of which `matches` are True
                T = tf.boolean_mask(T,Mgentile)             # (matches, 6, (genomeIDs, contigIDs, startPos, aa_seqlen))
                M = tf.boolean_mask(M,Mgentile)             # (matches, 6, T-k+1, U)
                Z = tf.boolean_mask(Z,Mgentile)             # (matches, 6, T-k+1, U)
                Idx = tf.boolean_mask(Idx, Mgentile)        # (matches, 6, T-k+1, U, 3)

                # manual broadcast of T to the correct shape
                T = tf.repeat(tf.expand_dims(T, 2), [M.shape[-1]], axis=2) # (matches, 6, U, (g, c, startPos, aaSeqlen))
                T = tf.repeat(tf.expand_dims(T, 2), [M.shape[-2]], axis=2) # (matches, 6, T-k+1, U, (g, c, s, l))
                
                # reduce to single match sites
                Idx = tf.boolean_mask(Idx, M)    # (sites, (f, r, u))
                T = tf.boolean_mask(T, M)        # (sites, (genomeID, contigID, startPos, aa_seqlen))
                Z = tf.boolean_mask(Z, M)        # (sites) where each entry is a score
                Z = tf.expand_dims(Z, -1)        # (sites, (score))
                R = tf.concat((Idx, T), axis=1 ) # (sites, (f, r, u, genomeID, contigID, startPos, aaSeqlen))

                fwdMask = tf.less(R[:,0], 3)         # (sites)
                rcMask = tf.greater_equal(R[:,0], 3) # (sites)
                rFwd = tf.boolean_mask(R, fwdMask)   # (fwdSites, (f, r, u, g, c, startPos, aaSeqlen, score))
                rRC = tf.boolean_mask(R, rcMask)     # ( rcSites, (f, r, u, g, c, startPos, aaSeqlen, score))
                zFwd = tf.boolean_mask(Z, fwdMask)   # (fwdSites, (score))
                zRC = tf.boolean_mask(Z, rcMask)     # ( rcSites, (score))

                # fwd case: p *= 3; p += tileStart
                posFwd = tf.multiply(rFwd[:,1], 3)   # (fwdSites), *3
                posFwd = tf.add(posFwd, rFwd[:,5])   # (fwdSites), add start
                sites = tf.concat([tf.expand_dims(rFwd[:,3], -1), 
                                   tf.expand_dims(rFwd[:,4], -1), 
                                   tf.expand_dims(posFwd, -1), 
                                   tf.expand_dims(rFwd[:,2], -1), 
                                   tf.expand_dims(rFwd[:,0], -1)], axis=1) # (fwdSites, (genomeID, contigID, pos, u, f))
                
                # rc case: p *= 3; p = rcStart - p - (k*3) + 1
                posRC = tf.multiply(rRC[:,1], 3)       # (rcSites), *3
                posRC = tf.subtract(rRC[:,5], posRC)   # (rcSites), start - p
                posRC = tf.subtract(posRC, (self.k*3)) # (rcSites), -(k*3)
                posRC = tf.add(posRC, 1)               # (rcSites), +1
                sitesRC = tf.concat([tf.expand_dims(rRC[:,3], -1), 
                                     tf.expand_dims(rRC[:,4], -1), 
                                     tf.expand_dims(posRC, -1), 
                                     tf.expand_dims(rRC[:,2], -1), 
                                     tf.expand_dims(rRC[:,0], -1)], axis=1) # (fwdSites, (g, c, pos, u, f))
                
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
            
        nlinks = None
        if calculateLinks:
            nunits = self.units if pIdx is None else 1
            for u in range(nunits):
                if pIdx is None:
                    umask = tf.equal(matches[:,3], u)
                    occs = tf.boolean_mask(matches, umask, axis=0) # get only matches of that unit
                else:
                    occs = matches
                    
                _, _, count = tf.unique_with_counts(occs[:,0])
                nlinks_u = tf.reduce_prod(count)
                if nlinks is None:
                    nlinks = nlinks_u
                else:
                    nlinks = tf.concat([nlinks, nlinks_u], axis=0)
        
        return matches, scores, nlinks


    
    def get_best_profile(self, ds):
        """ Return index of the profile that has the lowest loss """
        Ls = []
        for batch in ds:
            X = batch[0]        # (B, tilePerX, N, 6, tileSize, 21)
            posTrack = batch[1] # (B, tilePerX, N, 6, 4)
            assert len(X.shape) == 6, str(X.shape)
            assert posTrack.shape != (1, 0), str(posTrack.shape)+" -- use batch dataset with position tracking!"
            assert X.shape[0:4] == posTrack.shape[0:4], str(X.shape)+" != "+str(posTrack.shape)
            for b in range(X.shape[0]): # iterate samples in batch
                _, _, Z = self.call(X[b])      # Z: (ntiles, N, 6, tile_size-k+1, U)
                _, loss_by_unit = self.loss(Z) # (U)

                # (tilePerX, N, f) -> -1 if contig was exhausted -> False if exhausted -> 1 for valid contig, 0 else
                W = tf.cast(posTrack[b,:,:,:,0] != -1, tf.float32)
                W = tf.multiply(tf.reduce_sum(W), tf.ones(shape = [self.units], # weight for the means, shape (U)
                                                          dtype=tf.float32)) 
                Ls.append(tf.multiply(loss_by_unit, W).numpy()) # store weighted losses

                if tf.reduce_any( tf.math.is_nan(Z) ):
                    print("[DEBUG] >>> nan in Z")
                    #print("[DEBUG] >>> M:", M)
                    print("[DEBUG] >>> W:", W)
                    #print("[DEBUG] >>> W1:", W1)
                    #print("[DEBUG] >>> Ms:", Ms)
                    
        B = tf.reduce_mean(Ls, axis=0) # get overall lowest mean loss per profile
        
        # [DEBUG] track profile losses over time
        if self.P_report_plosshist is None:
            self.P_report_plosshist = tf.reshape(B, (1,-1))
        else:
            self.P_report_plosshist =  tf.concat([self.P_report_plosshist, tf.reshape(B, (1,-1))], axis=0)
        
        return tf.argmin(B), tf.reduce_min(B) # index, mean loss



    def getP(self):
        P = tf.nn.softmax(self.P_logit, axis=1, name="P")
        return P                                 # shape: (k, alphabet_size, U)



    def getP_report_raw(self):
        if len(self.P_report) == 0:
            return None

        P = tf.transpose(self.P_report, [1,2,0]) # shape: (k, alphabet_size, -1)
        return P
    


    def getP_report(self):
        if len(self.P_report) == 0:
            return None, None, None
            
        P = tf.nn.softmax(self.getP_report_raw(), axis=1, name="P_report")
        return P, self.P_report_thresold, self.P_report_loss
    


    def getP_report_whole(self):
        P = tf.transpose(self.P_report_whole, [1,2,0]) # shape: (k, alphabet_size, -1)
        P = tf.nn.softmax(P, axis=1, name="P_report_whole")
        return P, self.P_report_whole_score, self.P_report_whole_loss
    


    def getP_optimal(self, dsHelper, loss_threshold = 0):
        """ 
        Return a np array with profiles of length k, shape (k, alphabet_size, U*), 
          as well as a list of scores and losses with shape (U*,) respectively.

        k-profiles are extracted from "whole" (i.e. k+2*shift) profiles 
          that have a loss below `loss_threshold` but only if they are no edge cases
        """
        
        #pScores = self.max_profile_scores(ds_score)
        pLosses = self.min_profile_losses(dsHelper.getDataset())
        mask = tf.less_equal(pLosses, loss_threshold)
        P = tf.boolean_mask(self.P_logit, mask, axis=2)   # (k+2s, alphabet_size, -1)
        P = tf.nn.softmax(P, axis=1)
        U = P.shape[-1]
        
        # Extract k-profiles from P
        P2 = tf.expand_dims(P[0:self.k, :, :], -1) # (k, alphabet_size, U, 1) 
        for i in tf.range(1, 1+(2*self.s), dtype=tf.int32): # [0, 1, 2, ...]
            P2_i = tf.expand_dims(P[i:self.k+i, :, :], -1) # (k, alphabet_size, U, 1) 
            P2 = tf.concat([P2, P2_i], axis=-1)            # (k, alphabet_size, U, 2s+1)
            
        assert P2.shape == (self.k, self.alphabet_size, U, 1+(2*self.s)), \
            f"{P2.shape} != {(self.k, self.alphabet_size, U, 1+(2*self.s))}"
        losses = self.min_profile_losses(dsHelper.getDataset(), 
                                         otherP = tf.reshape(P2, (self.k, self.alphabet_size, -1)))
        scores = self.max_profile_scores(dsHelper.getDataset(), 
                                         otherP = tf.reshape(P2, (self.k, self.alphabet_size, -1)))
        losses = tf.reshape(losses, (U, 1+(2*self.s))) # (U, 2s+1)
        scores = tf.reshape(scores, (U, 1+(2*self.s))) # (U, 2s+1)
        
        bestShift = tf.math.argmax(scores, axis = 1)        # (U)
        scores = tf.gather(scores, bestShift, batch_dims=1) # (U)
        losses = tf.gather(losses, bestShift, batch_dims=1) # (U)
        #print("[DEBUG] >>> U:", U)
        #print("[DEBUG] >>> bestShift shape:", bestShift.shape)
        #print("[DEBUG] >>> gathered scores shape:", scores.shape)
        #print("[DEBUG] >>> gathered losses shape:", losses.shape)
        # exclude best shifts at edges
        shiftMask = tf.logical_not(tf.logical_or(tf.equal(bestShift, 0), tf.equal(bestShift, 2*self.s))) 
        #print("[DEBUG] >>> shiftMask shape:", shiftMask.shape)
        bestShift = tf.boolean_mask(bestShift, shiftMask, axis=0) # (U*)
        scores = tf.boolean_mask(scores, shiftMask, axis=0)
        losses = tf.boolean_mask(losses, shiftMask, axis=0)
        #print("[DEBUG] >>> masked bestShift shape:", bestShift.shape)
        #print("[DEBUG] >>> masked scores shape:", scores.shape)
        #print("[DEBUG] >>> masked losses shape:", losses.shape)
        #print("[DEBUG] >>> P2 shape:", P2.shape)
        P2 = tf.boolean_mask(P2, shiftMask, axis=2) 
        #print("[DEBUG] >>> masked P2 shape:", P2.shape)        
        P2 = tf.gather(tf.transpose(P2, [2,3,0,1]), indices=bestShift, batch_dims=1) # (U*, k, alphabet_size)
        #print("[DEBUG] >>> gathered P2 shape:", P2.shape)
        P2 = tf.transpose(P2, [1,2,0]) # (k, alphabet_size, U*)
        #print("[DEBUG] >>> transposed P2 shape:", P2.shape)
        
        return P2, scores, losses



    def getHistory(self):
        """ return training history """
        convHist = {} # convert tf list wrappers to numpy arrays
        for key in self.history:
            convHist[key] = np.array(self.history[key])

        return convHist
        
        

    def getR(self, otherP = None):
        """ Return shape is (k, alphabet_size, U).
            `otherP` must be _softmaxed_, don't pass the logits! """
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
            
        return R # shape: (k, alphabet_size, U)
    


    def getZ(self, X, otherP = None):
        """ Return shape is (ntiles, N, 6, tile_size-k+1, U) and R ((k, alphabet_size, U)). """
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
        """ Returns S, R, Z,
            shapes are (ntiles, N, U), (k, alphabet_size, U) and (ntiles, N, 6, tile_size-k+1, U). """
        Z, R = self.getZ(X, otherP)

        S = tf.reduce_max(Z, axis=[2,3])   # shape (ntiles, N, U)
        return S, R, Z



    def loss(self, Z, otherP = None):
        """ Returns the score and the loss per profile (shape (U)).
            Scores is the max loss over all tiles and frames, summed up for all genomes and profiles.
            Loss per profile is the softmax over all positions (tiles, frames) per genome and profile, maxed for each
               profile and summed over all genomes."""
        #print ("1st of loss calculation: Z.shape=", Z.shape)
        # shape of Z: ntiles x N x 6 x tile_size-k+1 x U 
        S = tf.reduce_max(Z, axis=[0,2,3]) # N x U
        score = tf.reduce_sum(S)
        
        if self.softmaxLoss:
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
        if self.experimentLoss:
            Z = tf.transpose(Z, [1,4,0,2,3]) # shape N x U x ntiles x 6 x tile_size-k+1
            Z = tf.reshape(Z, [Z.shape[0], Z.shape[1], -1]) # shape N x U x -1
            Zsm = tf.nn.softmax(self.gamma*Z, axis=-1) # softmax for each profile in each genome 
            Z = tf.math.multiply(Z, Zsm)
            loss_by_unit = -tf.math.reduce_max(Z, axis=-1) # best isolated match for each profile in each genome
            loss_by_unit = tf.math.reduce_sum(loss_by_unit, axis=0) # best isolated match of all genomes (not sum anymore)
            
        else:
            loss_by_unit = -tf.math.reduce_sum(S, axis=0)
            
        # L2 regularization
        P = self.P_logit if otherP is None else otherP
        L2 = tf.reduce_sum(tf.math.square(P), axis=[0,1]) # U
        L2 = tf.math.divide(L2, P.shape[0])
        L2 = tf.math.multiply(L2, self.l2)
        #print("[DEBUG] >>> Z.shape:", Z.shape, "loss_by_unit.shape:", loss_by_unit.shape, "L2.shape:", L2.shape)
        loss_by_unit = tf.math.add(loss_by_unit, L2)
        
        return score, loss_by_unit



    def max_profile_scores(self, ds, otherP = None):
        """ Return for each profile the best score at any position in the dataset (shape (U)). """
        U = self.units if otherP is None else otherP.shape[-1]
        scores = tf.ones([U], dtype=tf.float32) * -np.infty
        for batch, _ in ds:
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                S, _, _ = self.call(X, otherP)                            # shape (ntiles, N, U)
                scores = tf.maximum(tf.reduce_max(S, axis=(0,1)), scores) # shape (U)
                                    
        return scores
    


    def profile_losses(self, ds, otherP = None):
        """ Return for each profile and each batch the loss contribution (shape (U, x) 
              where x is number_of_batches * batch_size) """
        U = self.units if otherP is None else otherP.shape[-1]
        losses = tf.zeros([U, 0], dtype=tf.float32) # shape (U, 0)
        for batch, _ in ds:
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                _, _, Z = self.call(X, otherP)
                _, loss_by_unit = self.loss(Z, otherP) # shape (U)
                #losses += losses_by_unit
                losses = tf.concat([losses, tf.expand_dims(loss_by_unit, -1)], axis=1) # shape (U, x)
                                    
        return losses
    


    def min_profile_losses(self, ds, otherP = None):
        """ Sums up the loss of each profile at each batch in the dataset (shape (U))."""
        lossesPerBatch = self.profile_losses(ds, otherP)   # (U, x)
        losses = tf.reduce_sum(lossesPerBatch, axis=-1)    # (U)
        return losses
    


    @tf.function()
    def train_step(self, X):
        with tf.GradientTape() as tape:
            S, R, Z = self.call(X)
            score, loss_by_unit = self.loss(Z)
            # Mario's loss
            #loss = -score
            loss = tf.reduce_sum(loss_by_unit)
            
        grad = tape.gradient(loss, self.P_logit)
        self.opt.apply_gradients([(grad, self.P_logit)])
        
        return S, R, loss
        


    def train_classic(self, ds, steps_per_epoch, epochs, learning_rate=1., verbose=True):
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) # large learning rate is much faster
        Lb = []
        Smin, Smax = float('inf'), float('-inf')
        Rmin, Rmax = float('inf'), float('-inf')
        for i in range(epochs):
            steps = 0
            for batch, _ in ds:
                for X in batch:
                    S, R, loss = self.train_step(X)
                    
                    Lb.append(loss)
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
            self.history['learning rate'].append(learning_rate)
                    
            if verbose and (i%(25) == 0 or i==epochs-1):
                S, R, Z = self(X)
                score, _ = self.loss(Z)
                print(f"epoch {i:>5} score={score.numpy():.4f}" +
                      " max R: {:.3f}".format(tf.reduce_max(R).numpy()) +
                      " min R: {:.3f}".format((tf.reduce_min(R).numpy())))
                
            
    
    def train_reporting(self, dsHelper, steps_per_epoch, epochs,
                        profile_plateau=5, profile_plateau_dev=1, match_score_factor=0.8,
                        learning_rate=1., n_best_profiles=None, verbose=True, verbose_freq=100):
        """ epochs is the number of epochs to train if n_best_profiles is None, otherwise it's the max number
              of epochs to wait before a forced profile report """
        max_epochs = None if n_best_profiles is None else epochs
        
        learning_rate_init = float(learning_rate)
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_init) # large learning rate is much faster
        
        # [DEBUG] update first max_scores in tracking if desired
        if self.track_profiles is not None and len(self.track_profiles) > 0:
            assert len(self.tracking['max_score']) == 1, str(self.tracking)
            assert len(self.tracking['P']) == 1, str(self.tracking)
            self.tracking['max_score'][0] = self.max_profile_scores(dsHelper.getDataset(), otherP = self.tracking['P'][0])
        # [DEBUG]/

        def profileHistInit():
            return {
                'idx': np.ndarray([profile_plateau], dtype=np.int), 
                'score': np.ndarray([profile_plateau], dtype=np.int),
                'i': 0,
                'c': 0}
        
        def setLR(learning_rate):
            print("[DEBUG] >>> Setting learning rate to", learning_rate)
            self.opt.learning_rate.assign(learning_rate)
            
        def reduceLR(learning_rate, patience=5, factor=0.75):
            if len(self.history['loss']) > patience:
                lastmin = self.history['loss'][-(patience+1)]
                if not any([l < lastmin for l in self.history['loss'][-patience:]]):
                    print("[INFO] >>> Loss did not decrease for", patience, "epochs, reducing learning rate from", 
                          learning_rate, "to", factor*learning_rate)
                    learning_rate *= factor
                    setLR(learning_rate)
                    
            return learning_rate

        tstart = time()
        i = 0
        profileHist = profileHistInit()
        edgeCaseCounter = 0
        run = True
        while run:
            steps = 0
            Lb = []
            Smin, Smax = float('inf'), float('-inf')
            Rmin, Rmax = float('inf'), float('-inf')
            ds_train = dsHelper.getDataset(repeat = True)
            ds_eval = dsHelper.getDataset(withPosTracking = True)
            for batch, _ in ds_train:         # shape: (batchsize, ntiles, N, 6, tile_size, alphabet_size)
                for X in batch:            # shape: (ntiles, N, 6, tile_size, alphabet_size)
                    assert len(X.shape) == 5, str(X.shape)
                    #print("[DEBUG] >>> performing train_step")
                    S, R, loss = self.train_step(X)
                    #print("[DEBUG] >>> train_step performed")
                    Lb.append(loss)
                    Rmax = max(Rmax, tf.reduce_max(R).numpy())
                    Rmin = min(Rmin, tf.reduce_min(R).numpy())
                    Smax = max(Smax, tf.reduce_max(S).numpy())
                    Smin = min(Smin, tf.reduce_min(S).numpy())
                    
                steps += 1
                if steps >= steps_per_epoch:
                    break
                    
            p, s = self.get_best_profile(ds_eval)
            
            # [DEBUG]
            self.P_report_bestlosshist.append(s)
            self.P_report_bestlosshistIdx.append(p)
            
            # [DEBUG] track profiles
            if self.track_profiles is not None and len(self.track_profiles) > 0:
                Pt = tf.gather(self.getP(), self.track_profiles, axis=2)
                self.tracking['epoch'].append(i+1)
                self.tracking['P'].append(Pt)
                self.tracking['max_score'].append(self.max_profile_scores(dsHelper.getDataset(), otherP = Pt))
            
            #print("epoch", i, "best profile", p.numpy(), "with score", s.numpy())
            profileHist['idx'][profileHist['i']] = p
            profileHist['score'][profileHist['i']] = s.numpy()
            profileHist['i'] = 0 if profileHist['i']+1 == profile_plateau else profileHist['i']+1
            profileHist['c'] += 1

            self.history['loss'].append(np.mean(Lb))
            self.history['Rmax'].append(Rmax)
            self.history['Rmin'].append(Rmin)
            self.history['Smax'].append(Smax)
            self.history['Smin'].append(Smin)
            self.history['learning rate'].append(learning_rate)

            if max_epochs is not None and profileHist['c'] > max_epochs:
                print("[WARNING] >>> Could not find a good profile in time, force report of profile", p.numpy())
                edgeCase = self.profile_cleanup(p, match_score_factor, dsHelper)
                if edgeCase:
                    edgeCaseCounter += 1
                else:
                    edgeCaseCounter = 0
                    
                # reset training
                print("[DEBUG] >>> Resetting training")
                profileHist = profileHistInit()
                setLR(learning_rate_init)
                learning_rate = learning_rate_init

            else:
                if profileHist['c'] >= profile_plateau and all(profileHist['idx'] == p):
                    sd = np.std(profileHist['score'])
                    if sd <= profile_plateau_dev:
                        print("epoch", i, "best profile", p.numpy(), "with mean loss", s.numpy())
                        print("cleaning up profile", p.numpy())
                        
                        edgeCase = self.profile_cleanup(p, match_score_factor, dsHelper)
                        if edgeCase:
                            edgeCaseCounter += 1
                        else:
                            edgeCaseCounter = 0
                            
                        # [DEBUG] track profiles
                        if len(self.tracking['masking']) > 0 and self.tracking['masking'][-1]['after_epoch'] is None:
                            self.tracking['masking'][-1]['after_epoch'] = i
                            
                        # reset training
                        print("[DEBUG] >>> Resetting training")
                        profileHist = profileHistInit()
                        setLR(learning_rate_init)
                        learning_rate = learning_rate_init
                    
            if verbose and (i%(verbose_freq) == 0 or (n_best_profiles is None and i==epochs-1)):
                _, R, Z = self(X)
                _, loss_by_unit = self.loss(Z)
                tnow = time()
                print("epoch", i, "best profile", p.numpy(), "with mean loss", s.numpy())
                print(f"epoch {i:>5} profile loss sum = {tf.reduce_sum(loss_by_unit).numpy():.4f}" +
                      " max R: {:.3f}".format(tf.reduce_max(R).numpy()) +
                      " min R: {:.3f}".format((tf.reduce_min(R).numpy())) +
                      " time: {:.2f}".format(tnow-tstart)) 

            i += 1
            if n_best_profiles is not None:
                run = (len(self.P_report) < n_best_profiles)
                if edgeCaseCounter > 10:
                    print("[WARNING] >>> Training seems to be stuck in edge cases, aborting")
                    run = False
            else:
                run = (i < epochs)
                
            learning_rate = reduceLR(learning_rate)

                
                
    # old way without shift
    def profile_cleanup_bak(self, pIdx, threshold, dsHelper):
        losses = self.profile_losses(dsHelper.getDataset())[pIdx,:] # shape (x) where x is batch_size * number of batches
        # "remove" match sites from genomes
        matches, _, _ = self.get_profile_match_sites(dsHelper.getDataset(withPosTracking = True), 
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
        
        
        
    # works with shifts, pattern k-mer has the lowest loss, but repeat is still found 
    #   (usually first, although k-mer has bad loss)
    # leave it for now, as sorting by k-mer loss or introducing a filter 
    #   (for repeat profile, whole-loss is about 10*k-mer loss) is more cosmetic
    def profile_cleanup(self, pIdx, match_score_factor, dsHelper):
        """ Add profile at pIdx to report profiles, mask match sites, and get newly initialized profiles """
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
        
        # get best k-mer and report (unless it is the first or last k-mer when shift > 0)
        genomes = dsHelper.genomes
        losses = self.min_profile_losses(dsHelper.getDataset(), otherP = Pk)   # (U)
        #bestIdx = tf.math.argmin(losses, axis=0).numpy()
        scores = self.max_profile_scores(dsHelper.getDataset(), otherP = Pk)   # (U)
        bestIdx = tf.math.argmax(scores, axis=0).numpy()
        returnEdgeCase = False
        if bestIdx not in [0, Pk.shape[2]-1] or self.s == 0:
            # [DEBUG] get whole profile metrics
            whole_score = self.max_profile_scores(dsHelper.getDataset())[pIdx]
            whole_loss = self.min_profile_losses(dsHelper.getDataset())[pIdx]
            
            #threshold = match_score_factor * self.max_profile_scores(dsHelper.getDataset(), otherP = Pk)[bestIdx]
            threshold = match_score_factor * scores[bestIdx]
            minloss = tf.reduce_min(self.profile_losses(dsHelper.getDataset(), otherP = Pk)[bestIdx,:]).numpy()
            # "remove" match sites from genomes, site: (genomeID, contigID, pos, u)
            matches, _, nlinks = self.get_profile_match_sites(dsHelper.getDataset(withPosTracking = True), 
                                                              threshold, bestIdx, calculateLinks = True, otherP = Pk) 
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
            self.P_report_idx.append(pIdx)
            self.P_report_thresold.append(threshold)
            #self.P_report_loss.append(tf.reduce_mean(losses).numpy())
            self.P_report_loss.append(minloss)
            self.P_report_masked_sites.append(reportSites)
            self.P_report_nlinks.append(nlinks)
            
            self.P_report_whole.append(P[:,:,pIdx])
            self.P_report_whole_score.append(whole_score)
            self.P_report_whole_loss.append(whole_loss)
            
            #self.P_report_kmer_losses.append(losses)
            self.P_report_kmer_scores.append(scores)
            
            # [DEBUG] track profiles
            self.tracking['masking'].append({'P_report_masked_sites_index': len(self.P_report_masked_sites)-1,
                                             'after_epoch': None})
            
        else:
            print("Profile is an edge case, starting over")
            returnEdgeCase = True
            self.P_report_discarded.append(Pk_logit[:,:,bestIdx])
            if self.P_logit_init is not None:
                # otherwise get stuck with this profile
                self.P_logit_init[:,:,pIdx] = np.ones((self.P_logit_init.shape[0], self.P_logit_init.shape[1]), 
                                                      dtype=np.float32) * np.min(self.P_logit_init)

        # reset profiles
        if self.P_logit_init is None:
            self.P_logit.assign(self.seed_P_genome(genomes)) # completely new set of seeds
        else:
            self.P_logit.assign(self.P_logit_init)
            
        return returnEdgeCase

        
        
    def seed_P_ds_deprecated(self, ds):
        """
            Seed profiles P with profiles that represent units random positions in the input sequences.
            Positions are drawn uniformly from all positions in all sequences. 
            This is done with an old and neat trick online, so that the data has to be read only once.
        """
        rho = 2.0
        oneProfile_logit_like_Q = np.log(self.Q)
        # shape [k, alphabet_size, units]
        P_logit_init = self._getRandomProfiles() if self.P_logit_init is None else self.P_logit_init 
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
                    #elif np.random.choice(m) < self.units:
                    elif self.nprng.choice(m) < self.units:
                        #i = np.random.choice(self.units)
                        i = self.nprng.choice(self.units)
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
    
    
    
    def seed_P_genome(self, genomes, rho=2.0, sigma=0.1):
        if True:
            flatg = []
            for seqs in genomes:
                flatg.extend(seqs) # should be all references and thus no considerable memory overhead

            lensum = sum([len(s) for s in flatg])
            weights = [len(s)/lensum for s in flatg]
            weights = tf.nn.softmax(weights).numpy()
            #seqs = np.random.choice(len(flatg), self.units, replace=True, p=weights)
            seqs = self.nprng.choice(len(flatg), self.units, replace=True, p=weights)
            ks = self.k + (2*self.s) # trained profile width (k +- shift)

            oneProfile_logit_like_Q = np.log(self.Q)
            P_logit_init = np.zeros((ks, self.alphabet_size, self.units), dtype=np.float32)
            #rho = 2.0
            kdna = ks * 3
            for j in range(self.units):
                i = seqs[j] # seq index
                assert len(flatg[i]) > kdna, str(len(flatg[i]))
                #pos = np.random.choice(len(flatg[i])-kdna, 1)[0]
                pos = self.nprng.choice(len(flatg[i])-kdna, 1)[0]
                aa = dsg.sequence_translation(flatg[i][pos:pos+kdna])
                OH = dsg.oneHot(aa)
                assert OH.shape == (ks, self.alphabet_size), str(OH.shape)+" != "+str((ks, self.alphabet_size))
                #seed = rho * OH + oneProfile_logit_like_Q + np.random.normal(scale=sigma, size=OH.shape)
                seed = rho * OH + oneProfile_logit_like_Q + self.nprng.normal(scale=sigma, size=OH.shape)
                P_logit_init[:,:,j] = seed

            return P_logit_init
        else:
            return self._getRandomProfiles()