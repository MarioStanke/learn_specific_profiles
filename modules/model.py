#!/usr/bin/env python
import logging
import numpy as np
import tensorflow as tf
from time import time

from . import dataset
from . import ProfileFindingSetup
from . import sequtils as su



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
    def __init__(self, 
                 setup: ProfileFindingSetup.ProfileFindingTrainingSetup,
                 alphabet_size: int, 
                 rand_seed: int = None, **kwargs):
        """
        Set up model and most metaparamters
            Parameters:
                setup: ProfileFindingTrainingSetup object containing all metaparameters, data and initial profiles
                alphabet_size (int): number of possible symbols in profiles
                rand_seed (int): optional set a seed for tensorflow's rng
        """
        super().__init__(**kwargs)

        self.setup = setup
        assert self.setup.lossStrategy in ['softmax', 'score', 'experiment'], \
            "[ERROR] >>> loss must be either 'softmax', 'score' or 'experiment'"
        
        if rand_seed is not None:
            #print("[DEBUG] >>> setting tf global seed to", rand_seed)
            logging.debug(f"[model.__init__] >>> setting tf global seed to {rand_seed}")
            tf.random.set_seed(rand_seed)

        self.nprng = np.random.default_rng(rand_seed) # if rand_seed is None, unpredictable entropy is pulled from OS
        self.epsilon = 1e-6
        self.alphabet_size = alphabet_size
        self.softmaxLoss = (self.setup.lossStrategy == 'softmax')
        self.experimentLoss = (self.setup.lossStrategy == 'experiment')
        #print("[DEBUG] >>> using softmaxLoss:", self.softmaxLoss, "// using experimentLoss:", self.experimentLoss)
        logging.debug(f"[model.__init__] >>> using softmaxLoss: {self.softmaxLoss} // " + \
                      "using experimentLoss: {self.experimentLoss}")
        self.rand_seed = rand_seed
        
        self.history = {'loss': [],
                        'Rmax': [],
                        'Rmin': [],
                        'Smax': [],
                        'Smin': [],
                        'learning rate': []}
        
        if self.setup.initProfiles is None:
            self.setP_logit(self._getRandomProfiles())
            self.P_logit_init = None
        else:
            #print("[DEBUG] >>> Using initProfiles from training setup instead of random")
            logging.debug("[model.__init__] >>> Using initProfiles from training setup instead of random")
            self.P_logit_init = self.setup.initProfiles # shape: (k, alphabet_size, U)
            self.setP_logit(self.P_logit_init)

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
        if self.setup.trackProfiles is not None and len(self.setup.trackProfiles) > 0:
            Pt = tf.gather(self.getP(), self.setup.trackProfiles, axis=2)
            self.tracking['epoch'].append(0)
            self.tracking['P'].append(Pt)
            self.tracking['max_score'].append(tf.constant(np.zeros(Pt.shape[2]), dtype=tf.float32))

        if self.setup.phylo_t > 0.0:
            if self.setup.k != 20:
                #print("[WARNING] >>> phylo_t > 0 requires amino acid alphabet and k=20")
                logging.warning("[model.__init__] >>> phylo_t > 0 requires amino acid alphabet and k=20, " + \
                                f"not {self.setup.k}")
            else:
                Q = self.setup.phylo_t * tf.eye(20) # placeholder
                # above unit matrix should be replaced with the PAM1 rate matrix
                # read from a file, make sure the amino acid order is corrected for
                # tip: use the code from Felix at
                # https://github.com/Gaius-Augustus/learnMSA/blob/e0c283eb749f6307100ccb73dd371a3d2660baf9/learnMSA/msa_hmm/AncProbsLayer.py#L291
                # but check that the result is consistent with the literature
                self.A = tf.linalg.expm(Q)
    

    def setP_logit(self, P_logit_init):
        self.P_logit = tf.Variable(P_logit_init, trainable=True, name="P_logit") 
        
        
        
    def _getRandomProfiles(self):
        Q1 = tf.expand_dims(self.setup.data.Q, 0)
        Q2 = tf.expand_dims(Q1, -1)    # shape: (1, alphabet_size, 1)
        
        P_logit_like_Q = np.log(Q2.numpy())
        P_logit_init = P_logit_like_Q + self.nprng.normal(scale=4., size=[self.setup.k+(2*self.setup.s), 
                                                                          self.alphabet_size, 
                                                                          self.setup.U]).astype('float32')
        return P_logit_init           # shape: (self.k+(2*self.s), alphabet_size, U)
        
        

    # TODO: Rewrite this in order to make DNA input possible. Best is probably to remove position conversion altogether,
    # except from tiles to acutal sequence positions. Conversion from AA to DNA or vice versa should be handled in the
    # evaluation, not here. This should also simplify this function quite a bit.    
    # ATTENTION: Currently, position conversion is imprecise as frame shift is ignored. Will be fixed with aboves TODO.
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

            Returns:
                matches (tensor): tensors of shape (X, 5) where X is the number of found sites and the second dimension
                                  contains tuples with (genomeID, contigID, pos, u, f)
                scores (tensor): tensor of shape (X, 1) containing the scores of the found sites
                links (tensor): tensor of shape (X, 1) containing the number of possible links for each profile or
                                    None if calculateLinks is False
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
                posRC = tf.multiply(rRC[:,1], 3)             # (rcSites), *3
                posRC = tf.subtract(rRC[:,5], posRC)         # (rcSites), start - p
                posRC = tf.subtract(posRC, (self.setup.k*3)) # (rcSites), -(k*3)
                posRC = tf.add(posRC, 1)                     # (rcSites), +1
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
            nunits = self.setup.U if pIdx is None else 1
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
    


    def getMaskedSites(self, idx):
        """ Return the sites that were masked after reporting the profile at index idx. Idx refers to the index of the
            profile in the list of _reported_ profiles. A mapping of the initial profile index to the reported profile
            can be found in self.tracking['masking']. 
            
            Returns: list of sites (g, c, a) where g is the genome index, c is the contig index, and a is the start 
                     position of the masked site."""
        if len(self.P_report_masked_sites) == 0:
            return None # no reporting, so no masking
        
        assert idx < len(self.P_report_masked_sites), f"[ERROR] >>> {idx} >= {len(self.P_report_masked_sites)}"
        sites = []
        for site in self.P_report_masked_sites[idx]:
            _, g, c, a, _ = site
            #print(seq, g.numpy(), c.numpy(), a.numpy(), b.numpy())
            sites.append((g.numpy(), c.numpy(), a.numpy()))
        
        return sites


    
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
                W = tf.multiply(tf.reduce_sum(W), tf.ones(shape = [self.setup.U], # weight for the means, shape (U)
                                                          dtype=tf.float32)) 
                Ls.append(tf.multiply(loss_by_unit, W).numpy()) # store weighted losses

                if tf.reduce_any( tf.math.is_nan(Z) ):
                    #print("[DEBUG] >>> nan in Z")
                    logging.debug("[model.get_best_profile] >>> nan in Z")
                    #print("[DEBUG] >>> M:", M)
                    #print("[DEBUG] >>> W:", W)
                    logging.debug(f"[model.get_best_profile] >>> W: {W}")
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
        P1 = tf.nn.softmax(self.P_logit, axis=1, name="P")

        if self.setup.phylo_t == 0.0:
            # TODO: test phylo_t=0.0 does not change the results if the else clause is executed
            P2 = P1 # shortcut only for running time sake
        else:
            if self.setup.k != 20:
                #print("[WARNING] >>> phylo_t > 0 requires amino acid alphabet and k=20")
                logging.warning("[model.getP] >>> phylo_t > 0 requires amino acid alphabet and k=20, " + \
                                f"not {self.setup.k}")
                P2 = P1
            else:
                # assume that at each site a a 2 step random experiment is done
                # 1. an amino acid is drawn from distribution P1[a,:,u]
                # 2. it evolves for time phylo_t according to the Q matrix
                # The resulting distribution is P2[a,:,u]
                P2 = tf.einsum('abu,bc->acu', P1, self.A)
        return P2     # shape: (k, alphabet_size, U)



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
    


    def getP_optimal(self, loss_threshold = 0, lossStatistics = False):
        """ 
        loss_threshold: only consider profiles with a loss below this threshold
        lossStatistics: print some statistics about the loss distribution of all extracted profiles

        Return a np array with profiles of length k, shape (k, alphabet_size, U*), 
          as well as a list of scores and losses with shape (U*,) respectively.

        k-profiles are extracted from "whole" (i.e. k+2*shift) profiles 
          that have a loss below `loss_threshold` but only if they are no edge cases
        """
        
        #pScores = self.max_profile_scores(ds_score)
        pLosses = self.min_profile_losses(self.setup.getDataset())
        if lossStatistics:
            # print("[INFO] >>> min loss:", tf.reduce_min(pLosses).numpy())
            # print("[INFO] >>> max loss:", tf.reduce_max(pLosses).numpy())
            # print("[INFO] >>> mean loss:", tf.reduce_mean(pLosses).numpy())
            logging.info(f"[model.getP_optimal] >>> min loss: {tf.reduce_min(pLosses).numpy()}")
            logging.info(f"[model.getP_optimal] >>> max loss: {tf.reduce_max(pLosses).numpy()}")
            logging.info(f"[model.getP_optimal] >>> mean loss: {tf.reduce_mean(pLosses).numpy()}")

        mask = tf.less_equal(pLosses, loss_threshold)
        P = tf.boolean_mask(self.P_logit, mask, axis=2)   # (k+2s, alphabet_size, -1)
        P = tf.nn.softmax(P, axis=1)
        U = P.shape[-1]
        
        # Extract k-profiles from P
        P2 = tf.expand_dims(P[0:self.setup.k, :, :], -1)          # (k, alphabet_size, U, 1) 
        for i in tf.range(1, 1+(2*self.setup.s), dtype=tf.int32): # [0, 1, 2, ...]
            P2_i = tf.expand_dims(P[i:self.setup.k+i, :, :], -1)  # (k, alphabet_size, U, 1) 
            P2 = tf.concat([P2, P2_i], axis=-1)                   # (k, alphabet_size, U, 2s+1)
            
        assert P2.shape == (self.setup.k, self.alphabet_size, U, 1+(2*self.setup.s)), \
            f"{P2.shape} != {(self.setup.k, self.alphabet_size, U, 1+(2*self.setup.s))}"
        losses = self.min_profile_losses(self.setup.getDataset(), 
                                         otherP = tf.reshape(P2, (self.setup.k, self.alphabet_size, -1)))
        scores = self.max_profile_scores(self.setup.getDataset(), 
                                         otherP = tf.reshape(P2, (self.setup.k, self.alphabet_size, -1)))
        losses = tf.reshape(losses, (U, 1+(2*self.setup.s))) # (U, 2s+1)
        scores = tf.reshape(scores, (U, 1+(2*self.setup.s))) # (U, 2s+1)
        
        bestShift = tf.math.argmax(scores, axis = 1)        # (U)
        scores = tf.gather(scores, bestShift, batch_dims=1) # (U)
        losses = tf.gather(losses, bestShift, batch_dims=1) # (U)            
        #print("[DEBUG] >>> U:", U)
        #print("[DEBUG] >>> bestShift shape:", bestShift.shape)
        #print("[DEBUG] >>> gathered scores shape:", scores.shape)
        #print("[DEBUG] >>> gathered losses shape:", losses.shape)

        if self.setup.s > 0:
            # exclude best shifts at edges
            shiftMask = tf.logical_not(tf.logical_or(tf.equal(bestShift, 0), tf.equal(bestShift, 2*self.setup.s))) 
        else:
            # nothing to exclude
            shiftMask = tf.constant(True, shape=bestShift.shape)

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
        
        return P2, scores.numpy(), losses.numpy()



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
        Q1 = tf.expand_dims(self.setup.data.Q, 0)
        Q2 = tf.expand_dims(Q1, -1)
        # Limit the odds-ratio, to prevent problem with log(0).
        # Very bad matches of profiles are irrelevant anyways.
        ratio = tf.maximum(P/Q2, self.epsilon)
        R = tf.math.log(ratio)
        if tf.reduce_any(tf.math.is_nan(P)):
            # print("[DEBUG] >>> nan in P:", tf.reduce_any(tf.math.is_nan(P), axis=[0,1]), 
            #       tf.boolean_mask(P, tf.reduce_any(tf.math.is_nan(P), axis=[0,1]), axis=2))
            # print("[DEBUG] >>> Q:", self.setup.data.Q)
            logging.debug(f"[model.getR] >>> nan in P: {tf.reduce_any(tf.math.is_nan(P), axis=[0,1])} " + \
                          f"{tf.boolean_mask(P, tf.reduce_any(tf.math.is_nan(P), axis=[0,1]), axis=2)}")
            logging.debug(f"[model.getR] >>> Q: {self.setup.data.Q}")
            
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
            #print("[DEBUG] >>> nan in R")
            logging.debug("[model.getZ] >>> nan in R")
        if tf.reduce_any(tf.math.is_nan(X)):
            #print("[DEBUG] >>> nan in X")
            logging.debug("[model.getZ] >>> nan in X")
        if tf.reduce_any(tf.math.is_nan(Z)):
            #print("[DEBUG] >>> nan in Z")
            logging.debug("[model.getZ] >>> nan in Z")
        
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
            Zsm = tf.nn.softmax(self.setup.gamma*Z, axis=-1) # softmax for each profile in each genome 
            Z = tf.math.multiply(Z, Zsm)
            loss_by_unit = -tf.math.reduce_max(Z, axis=-1) # best isolated match for each profile in each genome
            loss_by_unit = tf.math.reduce_sum(loss_by_unit, axis=0) # best isolated match of all genomes (not sum anymore)
            
        else:
            loss_by_unit = -tf.math.reduce_sum(S, axis=0)
            
        # L2 regularization
        P = self.P_logit if otherP is None else otherP
        L2 = tf.reduce_sum(tf.math.square(P), axis=[0,1]) # U
        L2 = tf.math.divide(L2, P.shape[0])
        L2 = tf.math.multiply(L2, self.setup.l2)
        #print("[DEBUG] >>> Z.shape:", Z.shape, "loss_by_unit.shape:", loss_by_unit.shape, "L2.shape:", L2.shape)
        loss_by_unit = tf.math.add(loss_by_unit, L2)
        
        return score, loss_by_unit



    def max_profile_scores(self, ds, otherP = None):
        """ Return for each profile the best score at any position in the dataset (shape (U)). """
        U = self.setup.U if otherP is None else otherP.shape[-1]
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
        U = self.setup.U if otherP is None else otherP.shape[-1]
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
        


    def train_classic(self, ds, verbose=True):
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.setup.learning_rate) # large learning rate is much faster
        Lb = []
        Smin, Smax = float('inf'), float('-inf')
        Rmin, Rmax = float('inf'), float('-inf')
        for i in range(self.setup.epochs):
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
                if steps >= self.setup.steps_per_epoch:
                    break
                    
            self.history['loss'].append(np.mean(Lb))
            self.history['Rmax'].append(Rmax)
            self.history['Rmin'].append(Rmin)
            self.history['Smax'].append(Smax)
            self.history['Smin'].append(Smin)
            self.history['learning rate'].append(self.setup.learning_rate)
                    
            if verbose and (i%(25) == 0 or i==self.setup.epochs-1):
                S, R, Z = self(X)
                score, _ = self.loss(Z)
                # print(f"epoch {i:>5} score={score.numpy():.4f}" +
                #       " max R: {:.3f}".format(tf.reduce_max(R).numpy()) +
                #       " min R: {:.3f}".format((tf.reduce_min(R).numpy())))
                logging.info(f"[model.train_classic] >>> epoch {i:>5} score={score.numpy():.4f}" + \
                             " max R: {:.3f}".format(tf.reduce_max(R).numpy()) + \
                             " min R: {:.3f}".format((tf.reduce_min(R).numpy())))
                
            
    
    def train_reporting(self, verbose=True, verbose_freq=100):
        """ setup.epochs is the number of epochs to train if n_best_profiles is None, otherwise it's the max number
              of epochs to wait before a forced profile report """
        max_epochs = None if self.setup.n_best_profiles is None else self.setup.epochs
        
        learning_rate = self.setup.learning_rate # gets altered during training
        learning_rate_init = float(learning_rate)
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_init) # large learning rate is much faster
        
        # [DEBUG] update first max_scores in tracking if desired
        if self.setup.trackProfiles is not None and len(self.setup.trackProfiles) > 0:
            assert len(self.tracking['max_score']) == 1, str(self.tracking)
            assert len(self.tracking['P']) == 1, str(self.tracking)
            self.tracking['max_score'][0] = self.max_profile_scores(self.setup.getDataset(), 
                                                                    otherP = self.tracking['P'][0])
        # [DEBUG]/

        def profileHistInit():
            return {
                'idx': np.ndarray([self.setup.profile_plateau], dtype=int), 
                'score': np.ndarray([self.setup.profile_plateau], dtype=int),
                'i': 0,
                'c': 0}
        
        def setLR(learning_rate):
            #print("[DEBUG] >>> Setting learning rate to", learning_rate)
            logging.debug(f"[model.train_reporting.setLR] >>> Setting learning rate to {learning_rate}")
            self.opt.learning_rate.assign(learning_rate)
            
        def reduceLR(learning_rate):
            if len(self.history['loss']) > self.setup.lr_patience:
                lastmin = self.history['loss'][-(self.setup.lr_patience+1)]
                if not any([l < lastmin for l in self.history['loss'][-self.setup.lr_patience:]]):
                    # print("[INFO] >>> Loss did not decrease for", self.setup.lr_patience, 
                    #       "epochs, reducing learning rate from", learning_rate, "to", 
                    #       self.setup.lr_factor*learning_rate)
                    logging.info("[model.train_reporting.reduceLR] >>> Loss did not decrease for " + \
                                 f"{self.setup.lr_patience} epochs, reducing learning rate from {learning_rate} to " + \
                                 f"{self.setup.lr_factor*learning_rate}")
                    learning_rate *= self.setup.lr_factor
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
            ds_train = self.setup.getDataset(repeat = True)
            ds_eval = self.setup.getDataset(withPosTracking = True)
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
                if steps >= self.setup.steps_per_epoch:
                    break
                    
            p, s = self.get_best_profile(ds_eval)
            
            # [DEBUG]
            self.P_report_bestlosshist.append(s)
            self.P_report_bestlosshistIdx.append(p)
            
            # [DEBUG] track profiles
            if self.setup.trackProfiles is not None and len(self.setup.trackProfiles) > 0:
                Pt = tf.gather(self.getP(), self.setup.trackProfiles, axis=2)
                self.tracking['epoch'].append(i+1)
                self.tracking['P'].append(Pt)
                self.tracking['max_score'].append(self.max_profile_scores(self.setup.getDataset(), otherP = Pt))
            
            #print("epoch", i, "best profile", p.numpy(), "with score", s.numpy())
            profileHist['idx'][profileHist['i']] = p
            profileHist['score'][profileHist['i']] = s.numpy()
            profileHist['i'] = 0 if profileHist['i']+1 == self.setup.profile_plateau else profileHist['i']+1
            profileHist['c'] += 1

            self.history['loss'].append(np.mean(Lb))
            self.history['Rmax'].append(Rmax)
            self.history['Rmin'].append(Rmin)
            self.history['Smax'].append(Smax)
            self.history['Smin'].append(Smin)
            self.history['learning rate'].append(learning_rate)

            if max_epochs is not None and profileHist['c'] > max_epochs:
                #print("[WARNING] >>> Could not find a good profile in time, force report of profile", p.numpy())
                logging.warning("[model.train_reporting] >>> Could not find a good profile in time, " + \
                                f"force report of profile {p.numpy()}")
                edgeCase = self.profile_cleanup(p)
                if edgeCase:
                    edgeCaseCounter += 1
                else:
                    edgeCaseCounter = 0
                    
                # reset training
                #print("[DEBUG] >>> Resetting training")
                logging.debug("[model.train_reporting] >>> Resetting training")
                profileHist = profileHistInit()
                setLR(learning_rate_init)
                learning_rate = learning_rate_init

            else:
                if profileHist['c'] >= self.setup.profile_plateau and all(profileHist['idx'] == p):
                    sd = np.std(profileHist['score'])
                    if sd <= self.setup.profile_plateau_dev:
                        # print("epoch", i, "best profile", p.numpy(), "with mean loss", s.numpy())
                        # print("cleaning up profile", p.numpy())
                        logging.info(f"[model.train_reporting] >>> epoch {i} best profile {p.numpy()} with mean loss" +\
                                     f"{s.numpy()}")
                        logging.info(f"[model.train_reporting] >>> cleaning up profile {p.numpy()}")
                        
                        edgeCase = self.profile_cleanup(p)
                        if edgeCase:
                            edgeCaseCounter += 1
                        else:
                            edgeCaseCounter = 0
                            
                        # [DEBUG] track profiles
                        if len(self.tracking['masking']) > 0 and self.tracking['masking'][-1]['after_epoch'] is None:
                            self.tracking['masking'][-1]['after_epoch'] = i
                            
                        # reset training
                        #print("[DEBUG] >>> Resetting training")
                        logging.debug("[model.train_reporting] >>> Resetting training")
                        profileHist = profileHistInit()
                        setLR(learning_rate_init)
                        learning_rate = learning_rate_init
                    
            if verbose and (i%(verbose_freq) == 0 or (self.setup.n_best_profiles is None and i==self.setup.epochs-1)):
                _, R, Z = self(X)
                _, loss_by_unit = self.loss(Z)
                tnow = time()
                # print("epoch", i, "best profile", p.numpy(), "with mean loss", s.numpy())
                # print(f"epoch {i:>5} profile loss sum = {tf.reduce_sum(loss_by_unit).numpy():.4f}" +
                #       " max R: {:.3f}".format(tf.reduce_max(R).numpy()) +
                #       " min R: {:.3f}".format((tf.reduce_min(R).numpy())) +
                #       " time: {:.2f}".format(tnow-tstart)) 
                logging.info(f"[model.train_reporting] >>> epoch {i} best profile {p.numpy()} with mean loss " + \
                             f"{s.numpy()}")
                logging.info(f"[model.train_reporting] >>> epoch {i:>5} profile loss sum " + \
                             f"= {tf.reduce_sum(loss_by_unit).numpy():.4f}" + \
                             " max R: {:.3f}".format(tf.reduce_max(R).numpy()) + \
                             " min R: {:.3f}".format((tf.reduce_min(R).numpy())) + \
                             " time: {:.2f}".format(tnow-tstart)) 

            i += 1
            if self.setup.n_best_profiles is not None:
                run = (len(self.P_report) < self.setup.n_best_profiles)
                if edgeCaseCounter > 10:
                    #print("[WARNING] >>> Training seems to be stuck in edge cases, aborting")
                    logging.warning("[model.train_reporting] >>> Training seems to be stuck in edge cases, aborting")
                    run = False
            else:
                run = (i < self.setup.epochs)
                
            learning_rate = reduceLR(learning_rate)

                
                
    # works with shifts, pattern k-mer has the lowest loss, but repeat is still found 
    #   (usually first, although k-mer has bad loss)
    # leave it for now, as sorting by k-mer loss or introducing a filter 
    #   (for repeat profile, whole-loss is about 10*k-mer loss) is more cosmetic
    def profile_cleanup(self, pIdx):
        """ Add profile at pIdx to report profiles, mask match sites, and get newly initialized profiles """
        # get ks-mer, extract all k-mers, temporarily set k-mers as new profiles
        P = self.P_logit # shape: (k, alphabet_size, U)
        b = P[:,:,pIdx].numpy()
        Pk_logit = np.empty(shape=(self.setup.k, self.alphabet_size, (2*self.setup.s)+1), dtype=np.float32)
        for s in range(b.shape[0]-self.setup.k+1):
            Pk_logit[:,:,s] = b[s:(s+self.setup.k),:]
            
        Pk = tf.nn.softmax(Pk_logit, axis=1, name="Pk")
        #self.P_logit.assign(Pk)
        #Ubak = self.units
        #self.units = Pk.shape[2]
        
        # get best k-mer and report (unless it is the first or last k-mer when shift > 0)
        genomes = self.setup.genomes()
        losses = self.min_profile_losses(self.setup.getDataset(), otherP = Pk)   # (U)
        #bestIdx = tf.math.argmin(losses, axis=0).numpy()
        scores = self.max_profile_scores(self.setup.getDataset(), otherP = Pk)   # (U)
        bestIdx = tf.math.argmax(scores, axis=0).numpy()
        returnEdgeCase = False
        if bestIdx not in [0, Pk.shape[2]-1] or self.setup.s == 0:
            # [DEBUG] get whole profile metrics
            whole_score = self.max_profile_scores(self.setup.getDataset())[pIdx]
            whole_loss = self.min_profile_losses(self.setup.getDataset())[pIdx]
            
            #threshold = match_score_factor * self.max_profile_scores(self.setup.getDataset(), otherP = Pk)[bestIdx]
            threshold = self.setup.match_score_factor * scores[bestIdx]
            minloss = tf.reduce_min(self.profile_losses(self.setup.getDataset(), otherP = Pk)[bestIdx,:]).numpy()
            # "remove" match sites from genomes, site: (genomeID, contigID, pos, u)
            matches, _, nlinks = self.get_profile_match_sites(self.setup.getDataset(withPosTracking = True), 
                                                              threshold, bestIdx, calculateLinks = True, otherP = Pk) 
            #print("DEBUG >>> matches:", matches)
            reportSites = []
            for site in matches:
                #print("DEBUG >>> site:", site)
                g = site[0]
                c = site[1]
                a = site[2]
                b = a+(self.setup.k*3)
                
                # [DEBUG] report matched sites for each reported profile
                reportSites.append((su.sequence_translation(genomes[g][c][a:b].upper()), g, c, a, b))
                
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
            #print("Profile is an edge case, starting over")
            logging.info("[model.profile_cleanup] >>> Profile is an edge case, starting over")
            returnEdgeCase = True
            self.P_report_discarded.append(Pk_logit[:,:,bestIdx])
            if self.P_logit_init is not None:
                # otherwise get stuck with this profile
                self.P_logit_init[:,:,pIdx] = np.ones((self.P_logit_init.shape[0], self.P_logit_init.shape[1]), 
                                                      dtype=np.float32) * np.min(self.P_logit_init)

        # reset profiles
        if self.P_logit_init is None:
            self.P_logit.assign(self.seed_P_genome()) # completely new set of seeds
        else:
            self.P_logit.assign(self.P_logit_init)
            
        return returnEdgeCase

        
        
    def seed_P_genome(self):
        if True:
            flatg = []
            for seqs in self.setup.genomes():
                flatg.extend(seqs) # should be all references and thus no considerable memory overhead

            lensum = sum([len(s) for s in flatg])
            weights = [len(s)/lensum for s in flatg]
            weights = tf.nn.softmax(weights).numpy()
            #seqs = np.random.choice(len(flatg), self.units, replace=True, p=weights)
            seqs = self.nprng.choice(len(flatg), self.setup.U, replace=True, p=weights)
            ks = self.setup.k + (2*self.setup.s) # trained profile width (k +- shift)

            oneProfile_logit_like_Q = np.log(self.setup.data.Q)
            P_logit_init = np.zeros((ks, self.alphabet_size, self.setup.U), dtype=np.float32)
            #rho = 2.0
            kdna = ks * 3
            for j in range(self.setup.U):
                i = seqs[j] # seq index
                assert len(flatg[i]) > kdna, str(len(flatg[i]))
                #pos = np.random.choice(len(flatg[i])-kdna, 1)[0]
                pos = self.nprng.choice(len(flatg[i])-kdna, 1)[0]
                aa = su.sequence_translation(flatg[i][pos:pos+kdna])
                OH = dataset.oneHot(aa)
                assert OH.shape == (ks, self.alphabet_size), str(OH.shape)+" != "+str((ks, self.alphabet_size))
                #seed = rho * OH + oneProfile_logit_like_Q + np.random.normal(scale=sigma, size=OH.shape)
                seed = self.setup.rho * OH + oneProfile_logit_like_Q + self.nprng.normal(scale=self.setup.sigma, 
                                                                                         size=OH.shape)
                P_logit_init[:,:,j] = seed

            return P_logit_init
        else:
            return self._getRandomProfiles()
        

        
    # ==================================================================================================================

    # old way without shift
    def profile_cleanup_bak(self, pIdx, threshold):
        losses = self.profile_losses(self.setup.getDataset())[pIdx,:] # shape (x) where x is batch_size * number of batches
        # "remove" match sites from genomes
        matches, _, _ = self.get_profile_match_sites(self.setup.getDataset(withPosTracking = True), 
                                                     threshold, pIdx) # site: (genomeID, contigID, pos, u)
        #print("DEBUG >>> matches:", matches)
        genomes = self.setup.genomes()
        for site in matches:
            #print("DEBUG >>> site:", site)
            g = site[0]
            c = site[1]
            a = site[2]
            b = a+(self.setup.k*3)
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
        self.P_logit.assign(self.seed_P_genome()) # completely new set of seeds
        
        
        
    def seed_P_ds_deprecated(self, ds):
        """
            Seed profiles P with profiles that represent units random positions in the input sequences.
            Positions are drawn uniformly from all positions in all sequences. 
            This is done with an old and neat trick online, so that the data has to be read only once.
        """
        rho = 2.0
        oneProfile_logit_like_Q = np.log(self.setup.data.Q)
        # shape [k, alphabet_size, units]
        P_logit_init = self._getRandomProfiles() if self.P_logit_init is None else self.P_logit_init 
        ks = self.setup.k + (2*self.setup.s) # trained profile width (k +- shift)
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
                    if m < self.setup.U:
                        i = m
                    #elif np.random.choice(m) < self.units:
                    elif self.nprng.choice(m) < self.setupp.U:
                        #i = np.random.choice(self.units)
                        i = self.nprng.choice(self.setup.U)
                    if i >= 0:
                        # replace i-th profile with a seed profile build from the pattern starting at position j
                        # Seed is the background distribution, except the observed k-mer at pos j is more likely
                        seed = rho * PP[j:j+ks,:] + oneProfile_logit_like_Q
                        P_logit_init[:,:,i] = seed
                    m += 1

        return P_logit_init
    
    
    
    def seed_P_triplets_deprecated(self):
        #print("[WARNING] >>> Resetting setup.U to alphabet_size^3!")
        logging.warning("[model.seed_P_triplets_deprecated] >>> Resetting setup.U to alphabet_size^3!")
        self.setup.U = self.alphabet_size ** 3
        P_logit = np.zeros((self.setup.k+(2*self.setup.s), self.alphabet_size, self.setup.U), dtype=np.float32)
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
    
    
    
    