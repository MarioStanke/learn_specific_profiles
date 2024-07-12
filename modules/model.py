#!/usr/bin/env python
from dataclasses import dataclass
import logging
import numpy as np
import tensorflow as tf
from time import time

from . import ModelDataSet
from . import ProfileFindingSetup


# Helper Classes for Tracking, Training History and Reporting

class ProfilePerformanceCache:
    """ Track the best profiles found in the last cache_size epochs, needed for reporting during training. """
    def __init__(self, cache_size: int):
        assert cache_size > 0, "[ERROR] >>> cache_size must be a positive integer"
        self.cache_size = cache_size
        self.profile_idx = np.ones((cache_size,), dtype=int) * -1
        self.profile_score = np.ones((cache_size,), dtype=float) * -np.inf
        self.rotating_idx = 0
        self.epoch_count = 0

    def update(self, best_profile_idx, best_profile_score):
        self.profile_idx[self.rotating_idx] = best_profile_idx
        self.profile_score[self.rotating_idx] = best_profile_score
        self.epoch_count += 1
        self.rotating_idx += 1
        if self.rotating_idx == self.cache_size:
            self.rotating_idx = 0


class EpochHistory:
    """ Collect metrics during a single training step (i.e. epoch). Can later be reported to the overall history """
    def __init__(self):
        self.Smin = float('inf')
        self.Smax = float('-inf')
        self.Rmin = float('inf')
        self.Rmax = float('-inf')
        self.losses: list[float] = []

    def update(self, S: tf.Tensor, R: tf.Tensor, loss: float):
        self.Smin = min(self.Smin, tf.reduce_min(S).numpy())
        self.Smax = max(self.Smax, tf.reduce_max(S).numpy())
        self.Rmin = min(self.Rmin, tf.reduce_min(R).numpy())
        self.Rmax = max(self.Rmax, tf.reduce_max(R).numpy())
        self.losses.append(loss)


class TrainingHistory:
    """ Collect metrics during training. """
    def __init__(self, U): # TODO: <-- copilot code, review and adjust
        self._U: int = U
        self.loss: list[float] = []
        self.Rmin: list[float] = []
        self.Rmax: list[float] = []
        self.Smin: list[float] = []
        self.Smax: list[float] = []
        self.learning_rate: list[float] = []
        self.profile_losses = np.zeros((U, 0), dtype=np.float32) # track losses of all U profiles over time, (U, n)
        
    def update(self, epochHist: EpochHistory, learning_rate: float, profile_losses: np.ndarray):
        """ Add the metrics of a single epoch to the history. Required shapes of profile_losses: (U,) """
        assert profile_losses.shape == (self._U,), f"{profile_losses.shape=}"
        self.loss.append(np.mean(epochHist.losses))
        self.Rmin.append(epochHist.Rmin)
        self.Rmax.append(epochHist.Rmax)
        self.Smin.append(epochHist.Smin)
        self.Smax.append(epochHist.Smax)
        self.learning_rate.append(learning_rate)
        self.profile_losses = np.concatenate([self.profile_losses, np.expand_dims(profile_losses, -1)], axis=1)



class ProfileTracking:
    """ For debugging purposes, track desired profiles and their performance during training. """
    def __init__(self, tracking_ids: list, k: int, alphabet_size: int):
        self.tracking_ids = tracking_ids
        self.epoch = []
        self.P = np.zeros((k, alphabet_size, len(tracking_ids), 0), dtype=np.float32) # (k, alphabet_size, U', n_epochs)
        self.max_scores = np.zeros((len(tracking_ids), 0), dtype=np.float32)  # (U', n_epochs)
        self.mean_losses = np.zeros((len(tracking_ids), 0), dtype=np.float32) # (U', n_epochs)
        self.masked_sites = np.zeros((0, 6), dtype=np.int32)          # (n, 6)
        self.masked_sites_scores = np.zeros((0, 1), dtype=np.float32) # (n, 1)
        self.masked_sites_epoch = np.zeros((0, 1), dtype=np.int32)    # (n, 1), assigning the sites to the epoch


    def addEpoch(self, epoch: int, P: np.ndarray, max_scores: np.ndarray, mean_losses: np.ndarray, 
                 masked_sites: np.ndarray = None, masked_sites_scores: np.ndarray = None):
        """ Add a profile to the tracking. Required shapes:
            P: (k, alphabet_size, U'), max_scores: (U',), mean_losses: (U',), masked_sites: (n, 6), 
            masked_sites_scores: (n,) where U' == len(tracking_ids). """
        assert P.shape == (self.P.shape[:-1]), f"{P.shape=}"
        assert max_scores.shape == (len(self.tracking_ids),), f"{max_scores.shape=}"
        assert mean_losses.shape == (len(self.tracking_ids),), f"{mean_losses.shape=}"
        if masked_sites is not None or masked_sites_scores is not None:
            assert masked_sites is not None, f"{masked_sites=}"
            assert masked_sites_scores is not None, f"{masked_sites_scores=}"
            assert len(masked_sites.shape) == 2, f"{masked_sites.shape=}"
            assert masked_sites.shape[1] == 6, f"{masked_sites.shape=}"
            assert len(masked_sites_scores.shape) == 1, f"{masked_sites_scores.shape=}"
            assert masked_sites_scores.shape[0] == masked_sites.shape[0], \
                f"{masked_sites_scores.shape=} does not match {masked_sites.shape=}"

        self.epoch.append(epoch)
        self.P = np.concatenate([self.P, np.expand_dims(P, -1)], axis=3)
        self.max_scores = np.concatenate([self.max_scores, np.expand_dims(max_scores, -1)], axis=1)
        self.mean_losses = np.concatenate([self.mean_losses, np.expand_dims(mean_losses, -1)], axis=1)
        if masked_sites is not None:
            self.masked_sites = np.concatenate([self.masked_sites, masked_sites], axis=0)
            self.masked_sites_scores = np.concatenate([self.masked_sites_scores, 
                                                       np.expand_dims(masked_sites_scores, -1)], axis=0)
            self.masked_sites_epoch = np.concatenate([self.masked_sites_epoch, 
                                                      np.ones((masked_sites.shape[0], 1), dtype=np.int32) * epoch],
                                                     axis=0)



@dataclass
class ProfileReport:
    """ Collect reported profiles and related information. """
    k: int
    alphabet_size: int

    def __post_init__(self):
        self.epoch = []
        self.P = np.zeros((self.k, self.alphabet_size, 0), dtype=np.float32) # (k, alphabet_size, U_report)
        self.index = []
        self.threshold = []
        self.loss = []
        self.masked_sites = np.zeros((0, 6), dtype=np.int32)                 # (n, 6)
        self.masked_sites_scores = np.zeros((0,), dtype=np.float32)          # (n,)
        self.masked_sites_P_idx = np.zeros((0,), dtype=np.int32)             # (n,), assigning the sites to self.P idx
        self.nlinks = []


    def __len__(self):
        return self.P.shape[2]


    def addProfile(self, epoch: int, P: np.ndarray, index: int, threshold: float = None, loss: float = None, 
                   masked_sites: np.ndarray = None, masked_sites_scores: np.ndarray = None, 
                   nlinks: int = None):
        """ Add a profile to the report.
            Parameters:
                epoch: epoch when the profile was reported
                P: profile matrix of shape (k, alphabet_size)
                index: index of the profile in the original profile tensor
                threshold: score threshold used to identify matching sites
                loss: loss of the profile
                masked_sites: masked sites in the genomes, shape (n, 6)
                masked_sites_scores: scores of the masked sites, shape (n, 1)
                nlinks: number of links in the masked sites """
        assert P.shape == (self.k, self.alphabet_size), f"{P.shape=}"
        if masked_sites is not None or masked_sites_scores is not None:
            assert masked_sites is not None, f"{masked_sites=}"
            assert masked_sites_scores is not None, f"{masked_sites_scores=}"
            assert len(masked_sites.shape) == 2, f"{masked_sites.shape=}"
            assert masked_sites.shape[1] == 6, f"{masked_sites.shape=}"
            assert len(masked_sites_scores.shape) == 1, f"{masked_sites_scores.shape=}"
            assert masked_sites_scores.shape[0] == masked_sites.shape[0], \
                f"{masked_sites_scores.shape=} does not match {masked_sites.shape=}"
            
        self.epoch.append(epoch)
        self.P = np.concatenate([self.P, np.expand_dims(P, -1)], axis=2)
        P_idx = self.P.shape[2] - 1
        self.index.append(index)
        self.threshold.append(threshold)
        self.loss.append(loss)
        if masked_sites is not None:
            self.masked_sites = np.concatenate([self.masked_sites, masked_sites], axis=0)
            self.masked_sites_scores = np.concatenate([self.masked_sites_scores, masked_sites_scores], axis=0)
            self.masked_sites_P_idx = np.concatenate([self.masked_sites_P_idx, 
                                                      np.ones((masked_sites.shape[0],), dtype=np.int32) * P_idx], 
                                                     axis=0)

        self.nlinks.append(nlinks)


# ======================================================================================================================

# Model Class

# TODO: Rewrite ProfileFindingSetup to work with ModelDataSet (should be simpler)

class SpecificProfile(tf.keras.Model):
    def __init__(self, 
                 setup: ProfileFindingSetup.ProfileFindingTrainingSetup,
                 rand_seed: int = None, **kwargs):
        """
        Set up model and most metaparamters
            Parameters:
                setup: ProfileFindingTrainingSetup object containing metaparameters and initial profiles
                data: ModelDataSet object containing the data and data related metaparameters
                rand_seed (int): optional set a seed for tensorflow's rng
        """
        super().__init__(**kwargs)

        self.setup = setup
        self.data = setup.data
        self.opt = tf.keras.optimizers.Adam(learning_rate=float(self.setup.learning_rate))

        self.epsilon = 1e-6 # TODO: move that into setup!!

        # setting random seeds if desired
        if rand_seed is not None:
            logging.debug(f"[model.__init__] >>> setting tf global seed to {rand_seed}")
            tf.random.set_seed(rand_seed)

        self.nprng = np.random.default_rng(rand_seed) # if rand_seed is None, unpredictable entropy is pulled from OS

        # initializing profiles
        if self.setup.initProfiles is None:
            self.P_logit_init = None
            self.P_logit = tf.Variable(self._getRandomProfiles(), trainable=True, name="P_logit") 
        else:
            logging.debug("[model.__init__] >>> Using initProfiles from training setup instead of random")
            self.P_logit_init = self.setup.initProfiles # shape: (k, alphabet_size, U)
            self.P_logit = tf.Variable(self.P_logit_init, trainable=True, name="P_logit")

        # initialize phylogenetic model
        if self.setup.phylo_t > 0.0:
            if self.setup.k != 20:
                logging.warning("[model.__init__] >>> phylo_t > 0 requires amino acid alphabet and k=20, " + \
                                f"not {self.setup.k}")
                self.A = None
            else:
                Q = self.setup.phylo_t * tf.eye(20) # TODO: placeholder
                # above unit matrix should be replaced with the PAM1 rate matrix
                # read from a file, make sure the amino acid order is corrected for
                # tip: use the code from Felix at
                # https://github.com/Gaius-Augustus/learnMSA/blob/e0c283eb749f6307100ccb73dd371a3d2660baf9/learnMSA/msa_hmm/AncProbsLayer.py#L291
                # but check that the result is consistent with the literature
                self.A = tf.linalg.expm(Q)

        # initialize tracking and history
        self.history = TrainingHistory(self.setup.U)
        self.profile_report = ProfileReport(self.setup.k, self.data.alphabet_size())
        self.whole_profile_report = ProfileReport(self.setup.k+2*self.setup.s, self.data.alphabet_size())
        self.discarded_profile_report = ProfileReport(self.setup.k, self.data.alphabet_size())
        self.profile_tracking = ProfileTracking(self.setup.trackProfiles, 
                                                self.setup.k+2*self.setup.s, self.data.alphabet_size())

        # add initial profile tracking
        if len(self.setup.trackProfiles) > 0:
            Pt = tf.gather(self.getP(), self.setup.trackProfiles, axis=2)
            Pt_logit = tf.gather(self.P_logit, self.setup.trackProfiles, axis=2)
            scores = tf.reduce_max( self.get_profile_scores(self.data.getDataset(), Pt), axis=1 ).numpy()
            losses = self.get_mean_losses(self.data.getDataset(withPosTracking=True), Pt, Pt_logit).numpy()
            sites, site_scores = self.get_profile_match_sites(self.data.getDataset(withPosTracking=True), Pt, 
                                                              self.setup.match_score_factor * scores)
            self.profile_tracking.addEpoch(-1, Pt.numpy(), scores, losses, sites.numpy(), site_scores.numpy())

    

    def _getRandomProfiles(self):
        """ Returns a random profile matrix of shape (k+(2*s), alphabet_size, U). """
        Q1 = tf.expand_dims(self.data.Q, 0)
        Q2 = tf.expand_dims(Q1, -1) # shape: (1, alphabet_size, 1)
        
        P_logit_like_Q = np.log(Q2.numpy())
        P_logit_init = P_logit_like_Q + self.nprng.normal(scale=4., size=[self.setup.k+(2*self.setup.s), 
                                                                          self.data.alphabet_size(), 
                                                                          self.setup.U]).astype('float32')
        return P_logit_init # shape: (self.k+(2*self.s), alphabet_size, U)



    def getP(self):
        """ Returns softmaxed P (k, alphabet_size, U). """
        P1 = tf.nn.softmax(self.P_logit, axis=1, name="P")

        if self.setup.phylo_t == 0.0:
            # TODO: test phylo_t=0.0 does not change the results if the else clause is executed
            P2 = P1 # shortcut only for running time sake
        else:
            if self.A is None: # wrong k, don't use phylo_t
                P2 = P1
            else:
                # assume that at each site a a 2 step random experiment is done
                # 1. an amino acid is drawn from distribution P1[a,:,u]
                # 2. it evolves for time phylo_t according to the Q matrix
                # The resulting distribution is P2[a,:,u]
                P2 = tf.einsum('abu,bc->acu', P1, self.A)
        return P2 # shape: (k, alphabet_size, U)



    def getR(self, P):
        """ Returns R (k, alphabet_size, U). Argument `P` must be _softmaxed_, don't pass the logits! """
        Q1 = tf.expand_dims(self.data.Q, 0)
        Q2 = tf.expand_dims(Q1, -1)
        # Limit the odds-ratio, to prevent problem with log(0).
        # Very bad matches of profiles are irrelevant anyways.
        ratio = tf.maximum(P/Q2, self.epsilon)
        R = tf.math.log(ratio)
        if tf.reduce_any(tf.math.is_nan(P)):
            logging.debug(f"[model.getR] >>> nan in P: {tf.reduce_any(tf.math.is_nan(P), axis=[0,1])} " + \
                          f"{tf.boolean_mask(P, tf.reduce_any(tf.math.is_nan(P), axis=[0,1]), axis=2)}")
            logging.debug(f"[model.getR] >>> Q: {self.data.Q}")
            
        return R # shape: (k, alphabet_size, U)



    def getZ(self, X, P):
        """ Performs the convolution. Returns Z (ntiles, N, f, tile_size-k+1, U) and R (k, alphabet_size, U). 
            Argument `P` must be _softmaxed_, don't pass the logits! """
        R = self.getR(P)

        X1 = tf.expand_dims(X,-1) # 1 input channel   shape: (ntiles, N, 6, tile_size, alphabet_size, 1)
        R1 = tf.expand_dims(R,-2) # 1 input channel   shape: (k, alphabet_size, 1, U)

        # X1: (batch_shape (ntiles, N, 6), in_height (tile_size),     in_width (alphabet_size), in_channels (1))
        # R1:                                 (filter_height (k), filter_width (alphabet_size), in_channels (1), out_channels (U))
        # Z1: (batch_shape (ntiles, N, 6), tile_size-k+1, 1, U)
        Z1 = tf.nn.conv2d(X1, R1, strides=1,
                          padding='VALID', data_format="NHWC", name="Z")
        Z = tf.squeeze(Z1, 4) # remove input channel dimension   shape (ntiles, N, 6, tile_size-k+1, U)
        
        if tf.reduce_any(tf.math.is_nan(R)):
            logging.debug("[model.getZ] >>> nan in R")
        if tf.reduce_any(tf.math.is_nan(X)):
            logging.debug("[model.getZ] >>> nan in X")
        if tf.reduce_any(tf.math.is_nan(Z)):
            logging.debug("[model.getZ] >>> nan in Z")
        
        return Z, R



    def call(self, X, P):
        """ Returns S, R, Z; shapes are (ntiles, N, U), (k, alphabet_size, U) and (ntiles, N, f, tile_size-k+1, U). 
            Argument `P` must be _softmaxed_, don't pass the logits! """
        Z, R = self.getZ(X, P)

        S = tf.reduce_max(Z, axis=[2,3])   # shape (ntiles, N, U)
        return S, R, Z
    


    def lossfun(self, Z, P_logit):
        """ Returns the score (float) and the loss per profile (shape (U)).
            Scores is the max loss over all tiles and frames, summed up for all genomes and profiles.
            Loss per profile is the softmax over all positions (tiles, frames) per genome and profile, maxed for each
               profile and summed over all genomes. 
            Pass P_logit _instead of softmaxed P_, as the L2 regularization is weaker with value ranges close to 0."""
        # shape of Z: ntiles x N x f x tile_size-k+1 x U 
        S = tf.reduce_max(Z, axis=[0,2,3]) # N x U
        score = tf.reduce_sum(S)
        
        Z = tf.transpose(Z, [1,4,0,2,3]) # shape N x U x ntiles x f x tile_size-k+1
        Z = tf.reshape(Z, [Z.shape[0], Z.shape[1], -1]) # shape N x U x -1
        Zsm = tf.nn.softmax(self.setup.gamma*Z, axis=-1) # softmax for each profile in each genome 
        Z = tf.math.multiply(Z, Zsm)
        loss_by_unit = -tf.math.reduce_max(Z, axis=-1) # best isolated match for each profile in each genome (N x U)
        loss_by_unit = tf.math.reduce_sum(loss_by_unit, axis=0) # best isolated match of all genomes (U,)
            
        # L2 regularization
        # shape of P_logit: (k+2s, alphabet_size, U)
        L2 = tf.reduce_sum(tf.math.square(P_logit), axis=[0,1]) # U
        L2 = tf.math.divide(L2, P_logit.shape[0])
        L2 = tf.math.multiply(L2, self.setup.l2)
        loss_by_unit = tf.math.add(loss_by_unit, L2)      # U

        return score, loss_by_unit
    
    

    @tf.function()
    def train_step(self, X):
        with tf.GradientTape() as tape:
            S, R, Z = self.call(X, self.getP())
            score, loss_by_unit = self.lossfun(Z, self.P_logit) # TODO: what P to pass to loss? In old version, this switches (otherP is usually softmaxed, but here it's the logits)
            # Mario's loss
            #loss = -score
            loss = tf.reduce_sum(loss_by_unit)
            
        grad = tape.gradient(loss, self.P_logit)
        self.opt.apply_gradients([(grad, self.P_logit)])
        
        return S, R, loss
    


    def train(self, verbose=True, verbose_freq=100):
        """ setup.epochs is the number of epochs to train if n_best_profiles is None, otherwise it's the max number
              of epochs to wait before a forced profile report """
        
        def setLR(learning_rate):
            logging.debug(f"[model.train.setLR] >>> Setting learning rate to {learning_rate}")
            self.opt.learning_rate.assign(learning_rate)

        max_epochs = None if self.setup.n_best_profiles is None else self.setup.epochs
        learning_rate = self.setup.learning_rate # gets altered during training
        setLR(learning_rate) # reset learning rate to initial value for safety

        # start training loop
        training_start_time = time()
        epoch_count = 0
        profilePerfCache = ProfilePerformanceCache(self.setup.profile_plateau)
        edgecase_count = 0
        run = True
        while run:
            # run an epoch
            steps = 0
            ds_train = self.data.getDataset(repeat = True)
            epochHist = EpochHistory()
            for batch, _ in ds_train: # shape: (batchsize, ntiles, N, f, tile_size, alphabet_size)
                for X in batch:       # shape: (ntiles, N, f, tile_size, alphabet_size)
                    assert len(X.shape) == 5, str(X.shape)
                    S, R, loss = self.train_step(X)
                    epochHist.update(S, R, loss.numpy())
                    
                steps += 1
                if steps >= self.setup.steps_per_epoch:
                    break
                    
            mean_losses = self.get_mean_losses(self.data.getDataset(withPosTracking = True), 
                                               self.getP(), self.P_logit) # (U)
            best_profile = tf.argmin(mean_losses).numpy()
            best_profile_mean_loss = tf.reduce_min(mean_losses).numpy()
            
            # write history and tracking
            profilePerfCache.update(best_profile, best_profile_mean_loss)
            self.history.update(epochHist, learning_rate, mean_losses.numpy())
            if len(self.profile_tracking.tracking_ids) > 0:
                Pt = tf.gather(self.getP(), self.profile_tracking.tracking_ids, axis=2)
                scores = tf.reduce_max( self.get_profile_scores(self.data.getDataset(), P = Pt), axis=1 ).numpy()
                losses = tf.gather(mean_losses, self.profile_tracking.tracking_ids, axis=0).numpy()
                sites, site_scores = self.get_profile_match_sites(self.data.getDataset(withPosTracking = True), Pt, 
                                                                  self.setup.match_score_factor * scores)
                self.profile_tracking.addEpoch(epoch_count, Pt.numpy(), scores, losses, 
                                               sites.numpy(), site_scores.numpy())

            # check if a profile can be reported and report it
            if profilePerfCache.epoch_count >= self.setup.profile_plateau \
                                                                  and all(profilePerfCache.profile_idx == best_profile):
                stdev = np.std(profilePerfCache.profile_score)
                if stdev <= self.setup.profile_plateau_dev:
                    logging.info(f"[model.train] >>> epoch {epoch_count} best profile " \
                                    + f"{best_profile} with mean loss {best_profile_mean_loss}")
                    logging.info(f"[model.train] >>> cleaning up profile {best_profile}")
                    
                    edgecase = self.profile_cleanup(best_profile, epoch_count)
                    edgecase_count = edgecase_count+1 if edgecase else 0 # increase or reset edgecase count
                        
                    # reset training
                    logging.debug("[model.train] >>> Resetting training")
                    profilePerfCache = ProfilePerformanceCache(self.setup.profile_plateau)
                    learning_rate = self.setup.learning_rate
                    setLR(learning_rate)

            # if no profile has been found for too long, force report the current best
            elif max_epochs is not None and profilePerfCache.epoch_count > max_epochs:
                logging.warning("[model.train] >>> Could not find a good profile in time, " + \
                                f"force report of profile {best_profile}")
                edgecase = self.profile_cleanup(best_profile, epoch_count)
                edgecase_count = edgecase_count+1 if edgecase else 0 # increase or reset edgecase count
                    
                # reset training
                logging.debug("[model.train] >>> Resetting training")
                profilePerfCache = ProfilePerformanceCache(self.setup.profile_plateau)
                learning_rate = self.setup.learning_rate
                setLR(learning_rate)
                
            # log training progress in certain steps
            if verbose and (epoch_count % (verbose_freq) == 0 \
                            or (self.setup.n_best_profiles is None and epoch_count == self.setup.epochs-1)):
                tnow = time()
                losses, _ = self.get_profile_losses(self.data.getDataset(withPosTracking=True), 
                                                    self.getP(), self.P_logit)
                logging.info(f"[model.train] >>> epoch {epoch_count} best profile {best_profile} " \
                             + f"with mean loss {best_profile_mean_loss}")
                logging.info(f"[model.train] >>> epoch {epoch_count:>5} sum of profile tile losses " + \
                             f"= {tf.reduce_sum(losses).numpy():.4f}," + \
                             f" max R: {epochHist.Rmax:.3f}, min R: {epochHist.Rmin:.3f}," + \
                             f" time: {tnow-training_start_time:.2f}s") 

            # check if learning rate should decrease
            if len(self.history.loss) > self.setup.lr_patience:
                lastmin = self.history.loss[-(self.setup.lr_patience+1)] # loss before the last lr_patience epochs
                if not any([l < lastmin for l in self.history.loss[-self.setup.lr_patience:]]):
                    logging.info("[model.train_reporting.reduceLR] >>> Loss did not decrease for " + \
                                 f"{self.setup.lr_patience} epochs, reducing learning rate from {learning_rate} to " + \
                                 f"{self.setup.lr_factor*learning_rate}")
                    learning_rate *= self.setup.lr_factor
                    setLR(learning_rate)

            # determine if training should continue
            epoch_count += 1
            if self.setup.n_best_profiles is not None:
                run = (len(self.profile_report) < self.setup.n_best_profiles)
                if edgecase_count > 10:
                    logging.warning("[model.train_reporting] >>> Training seems to be stuck in edge cases, aborting")
                    run = False
            else:
                run = (epoch_count < self.setup.epochs)
                


    def get_profile_losses(self, ds, P, P_logit):
        """ Argument `P` must be _softmaxed_, P_logit is the logits of P (i.e. before softmaxing)!
            Returns a tensor of losses for each tile of shape (U, x) where x is number_of_batches * batch_size,
            and a tensor of weights of the same shape (U, x): In each batch <x>, the weight for all profiles <U> is the
            same; the weights are 1 if all tiles in all genomes and frames are valid, or smaller if some where 
            exhausted. The weight tensor can be used to compute a weighted mean loss per profile. """
        U = P.shape[-1]
        losses = tf.zeros([U, 0], dtype=tf.float32) # shape (U, 0)
        weights = tf.zeros([U, 0], dtype=tf.float32) # shape (U, 0)
        for batch in ds:
            X = batch[0]        # (B, tilePerX, N, f, tileSize, 21)
            posTrack = batch[1] # (B, tilePerX, N, f, 4)
            assert len(X.shape) == 6, str(X.shape)
            assert posTrack.shape != (1, 0), str(posTrack.shape)+" -- use batch dataset with position tracking!"
            assert X.shape[0:4] == posTrack.shape[0:4], f"{X.shape=} != {posTrack.shape=}"
            ntiles = np.prod(posTrack.shape[1:4]) # tilesPerX * N * f
            for b in range(X.shape[0]): # iterate samples in batch
                _, _, Z = self.call(X[b], P)               # Z: (ntiles, N, f, tile_size-k+1, U)
                _, loss_by_unit = self.lossfun(Z, P_logit) # (U)

                # (tilePerX, N, f) -> -1 if tile was exhausted -> False if exhausted -> 1 for valid tile, else 0
                W = tf.cast(posTrack[b,:,:,:,0] != -1, tf.float32) # binary mask for valid tiles, (tilePerX, N, f)
                W = tf.reduce_sum(W) / ntiles # weight for the tile, scalar
                W = tf.broadcast_to(W, (U, 1)) # weight for the tile, (U, 1)
                
                losses = tf.concat([losses, tf.expand_dims(loss_by_unit, -1)], axis=1)
                weights = tf.concat([weights, W], axis=1)
            
                if tf.reduce_any( tf.math.is_nan(Z) ):
                    logging.debug("[model.get_profile_losses] >>> nan in Z")
                    logging.debug(f"[model.get_profile_losses] >>> W: {W}")

        return losses, weights



    def get_mean_losses(self, ds, P, P_logit):
        """ A wrapper around get_profile_losses that returns the weighted mean loss per profile. 
            Argument `P` must be _softmaxed_, P_logit is the logits of P (i.e. before softmaxing)! 
            Argument `ds` must be a dataset with position tracking. """
        losses, weights = self.get_profile_losses(ds, P, P_logit)
        return tf.reduce_mean( tf.multiply(losses, weights), axis=1 ) # (U)
    


    def get_profile_scores(self, ds, P):
        """ Return for each profile the max score reached per batch, shape (U, x) where x is 
            number_of_batches * batch_size.
            Argument `P` must be _softmaxed_, don't pass the logits! """
        U = P.shape[-1]
        scores = None
        for batch, _ in ds:
            for X in batch:
                assert len(X.shape) == 5, str(X.shape)
                S, _, _ = self.call(X, P)        # shape (ntiles, N, U)
                S = tf.reduce_max(S, axis=(0,1)) # shape (U)
                if scores is None:
                    scores = tf.expand_dims(S, -1)
                else:
                    scores = tf.concat([scores, tf.expand_dims(S, -1)], axis=1)
                                    
        return scores
    


    def profile_cleanup(self, pIdx: int, epoch: int):
        """ Add profile at pIdx to report profiles, mask match sites, and get newly initialized profiles """
        # get k+2s-mer, extract all k-mers, temporarily set k-mers as new profiles
        P = self.P_logit # shape: (k+2s, alphabet_size, U)
        b = P[:,:,pIdx].numpy()
        Pk_logit = np.empty(shape=(self.setup.k, self.data.alphabet_size(), (2*self.setup.s)+1), dtype=np.float32)
        for s in range(b.shape[0]-self.setup.k+1):
            Pk_logit[:,:,s] = b[s:(s+self.setup.k),:]
            
        Pk = tf.nn.softmax(Pk_logit, axis=1, name="Pk") # shape: (k, alphabet_size, 2s+1 -> U')
        
        # get best k-mer and report (unless it is the first or last k-mer when shift > 0)
        scores = tf.reduce_max( self.get_profile_scores(self.data.getDataset(), P = Pk), axis=1 )   # (U', x) -> (U')
        bestIdx = tf.math.argmax(scores, axis=0).numpy()
        threshold = self.setup.match_score_factor * scores.numpy()[bestIdx]
        losses, _ = self.get_profile_losses(self.data.getDataset(withPosTracking=True), Pk, Pk_logit)
        minloss = tf.reduce_min(losses[bestIdx,:]).numpy()
        sites, sitescores = self.get_profile_match_sites(self.data.getDataset(withPosTracking = True), 
                                                         Pk, threshold, bestIdx)
        if bestIdx not in [0, Pk.shape[2]-1] or self.setup.s == 0:
            returnEdgeCase = False
            # report the best k-profile
            self.profile_report.addProfile(epoch, Pk[:,:,bestIdx].numpy(), pIdx, threshold, minloss, 
                                           sites.numpy(), sitescores.numpy())
        
            # "remove" match sites from genomes, site: <genomeIdx, contigIdx, frameIdx, tileStartPos, T-k+1_idx, U_idx>
            for site in sites:
                if all(site.numpy()[:4] == [-1, -1, -1, -1]):
                    logging.warning(f"[model.profile_cleanup] >>> Attempted to mask {site=} in exhausted tile, " \
                                    +"skipping.")
                    continue

                matchseq = self.data.softmask(genome_idx=site[0].numpy(), 
                                              sequence_idx=site[1].numpy(),
                                              frame_idx=site[2].numpy(), 
                                              start_pos=site[3].numpy()+site[4].numpy(), 
                                              masklen=self.setup.k)
                if len(matchseq) != self.setup.k:
                    logging.warning(f"[model.profile_cleanup] >>> Match sequence has wrong length: {len(matchseq)}" \
                                    + f", expected {self.setup.k}. Site {site} seems out of bounds")
                    
            # for debugging purpose, report the whole k+2s-profile as well
            whole_scores = tf.reduce_max( self.get_profile_scores(self.data.getDataset(), self.getP()), axis=1 ).numpy()
            whole_losses, _ = self.get_profile_losses(self.data.getDataset(withPosTracking=True), 
                                                      self.getP(), self.P_logit)
            self.whole_profile_report.addProfile(epoch, self.getP().numpy()[:,:,pIdx], pIdx, 
                                                 self.setup.match_score_factor * whole_scores[pIdx],
                                                 tf.reduce_min(whole_losses[pIdx,:]).numpy())
            
        else:
            returnEdgeCase = True
            logging.info("[model.profile_cleanup] >>> Profile is an edge case, starting over")
            self.discarded_profile_report.addProfile(epoch, Pk[:,:,bestIdx].numpy(), pIdx, threshold, minloss, 
                                                     sites.numpy(), sitescores.numpy())
            if self.P_logit_init is not None:
                # otherwise get stuck with this profile
                self.P_logit_init[:,:,pIdx] = np.ones((self.P_logit_init.shape[0], self.P_logit_init.shape[1]), 
                                                      dtype=np.float32) * np.min(self.P_logit_init)
            
        # reset profiles
        if self.P_logit_init is None:
            self.P_logit.assign(self._getRandomProfiles()) # initially, this called self.seed_P_genome() where the genomes were sampled for seeds
        else:
            self.P_logit.assign(self.P_logit_init)
            
        return returnEdgeCase
    


    def get_profile_match_sites(self, ds, P, score_threshold, pIdx: int = None):
        """
        Get sites in the dataset where either all or a specific profile match according to a score threshold
            Parameters:
                ds: tf dataset
                P: profile tensor, shape (k[+2s], alphabet_size, U), needs to be softmaxed! Don't pass the logits!
                score_threshold (float or tensor): matching sites need to achieve at least this score
                pIdx (int): optional index of a single profile, if given only matching sites of that profile are 
                              reported
                
            Returns:
                sites (tensor): tensor of shape (X, 6) where X is the number of found sites and the second dimension
                                contains tuples with (genomeIdx, contigIdx, frameIdx, tileStartPos, tilePos, profileIdx)
                scores (tensor): tensor of shape (X, 1) containing the scores of the found sites
        """        
        score_threshold = tf.convert_to_tensor(score_threshold, dtype=tf.float32)
        assert score_threshold.shape in [(), (P.shape[-1])], f"{score_threshold=}, {score_threshold.shape=}"
            
        sites = None
        scores = None
        for batch in ds:
            X_b = batch[0]        # (B, tilePerX, N, f, tileSize, alphabetSize)
            posTrack_b = batch[1] # (B, tilePerX, N, f, <genomeIdx, contigIdx, frameIdx, TileStartPos>)
            assert len(X_b.shape) == 6, str(X_b.shape)
            assert posTrack_b.shape != (1, 0), f"{posTrack_b.shape=} -- use batch dataset with position tracking!"
            assert X_b.shape[0:4] == posTrack_b.shape[0:4], f"{X_b.shape} != {posTrack_b.shape}"
            for b in range(X_b.shape[0]): # iterate samples in batch
                # get profile match scores, i.e. the sum of the element-wise multiplication of each profile 
                #   at each sequence position in X --> Z
                X = X_b[b]                # (tilePerX, N, f, tileSize, alphabetSize)
                posTrack = posTrack_b[b]  # (tilePerX, N, f, <genomeIdx, contigIdx, frameIdx, TileStartPos>)
                _, _, Z = self.call(X, P) # (tilePerX, N, f, T-k+1, U)
                if pIdx is not None:
                    Z = Z[:,:,:,:,pIdx:(pIdx+1)] # only single profile, but keep dimensions

                # identify matches, i.e. match score >= score_threshold
                M = tf.greater_equal(Z, score_threshold) # (tilesPerX, N, f, T-k+1, U)

                # index tensor -> 2D tensor with shape (sites, 5) where each row is a match and the columns are indices:
                I = tf.cast(tf.where(M), tf.int32)       # (sites, <tilesPerX_idx, N_idx, f_idx, T-k+1_idx, U_idx>)

                # build the sites and scores tensors (tensorflow.org/versions/r2.10/api_docs/python/tf/gather_nd)
                _scores = tf.gather_nd(Z, I)                  # (sites, <score>)
                _sites = tf.gather_nd(posTrack, I[:,:3])      # (sites, <g,c,f,tspos>)
                _sites = tf.concat([_sites, I[:,3:]], axis=1) # (sites, <g,c,f,tspos,T-k+1_idx,U_idx>)

                if sites is None:
                    sites = _sites
                    scores = _scores
                else:
                    sites = tf.concat([sites, _sites], axis=0)
                    scores = tf.concat([scores, _scores], axis=0)

        if sites is None:
            return tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.float32)
        
        return sites, scores



    # TODO: rewrite

    def get_optimal_P(self, loss_threshold = 0, loss_statistics = False):
        """ 
        loss_threshold: only consider profiles with a loss below this threshold
        loss_statistics: print some statistics about the loss distribution of all extracted profiles

        Return a np array with profiles of length k, shape (k, alphabet_size, U*), 
          as well as a list of scores and losses with shape (U*,) respectively.

        k-profiles are extracted from "whole" (i.e. k+2*shift) profiles 
          that have a loss below `loss_threshold` but only if they are no edge cases
        """
        
        raise NotImplementedError("This function is not yet implemented. Training is always performed with reporting.")
        
        # # TODO: figure out if we want mean loss or full loss tensors

        # #pScores = self.max_profile_scores(ds_score)
        # #pLosses = self.min_profile_losses(self.setup.getDataset())
        # losses = self.get_profile_losses(self.setup.getDataset(), self.getP(), self.P_logit)
        # if loss_statistics:
        #     logging.info(f"[model.getP_optimal] >>> overall min loss: {tf.reduce_min(losses).numpy()}")
        #     logging.info(f"[model.getP_optimal] >>> overall max loss: {tf.reduce_max(losses).numpy()}")
        #     logging.info(f"[model.getP_optimal] >>> overall mean loss: {tf.reduce_mean(losses).numpy()}")

        # mask = tf.less_equal(pLosses, loss_threshold)
        # P = tf.boolean_mask(self.P_logit, mask, axis=2)   # (k+2s, alphabet_size, -1)
        # P = tf.nn.softmax(P, axis=1)
        # U = P.shape[-1]
        
        # # Extract k-profiles from P
        # P2 = tf.expand_dims(P[0:self.setup.k, :, :], -1)          # (k, alphabet_size, U, 1) 
        # for i in tf.range(1, 1+(2*self.setup.s), dtype=tf.int32): # [0, 1, 2, ...]
        #     P2_i = tf.expand_dims(P[i:self.setup.k+i, :, :], -1)  # (k, alphabet_size, U, 1) 
        #     P2 = tf.concat([P2, P2_i], axis=-1)                   # (k, alphabet_size, U, 2s+1)
            
        # assert P2.shape == (self.setup.k, self.alphabet_size, U, 1+(2*self.setup.s)), \
        #     f"{P2.shape} != {(self.setup.k, self.alphabet_size, U, 1+(2*self.setup.s))}"
        # losses = self.min_profile_losses(self.setup.getDataset(), 
        #                                  otherP = tf.reshape(P2, (self.setup.k, self.alphabet_size, -1)))
        # scores = self.max_profile_scores(self.setup.getDataset(), 
        #                                  otherP = tf.reshape(P2, (self.setup.k, self.alphabet_size, -1)))
        # losses = tf.reshape(losses, (U, 1+(2*self.setup.s))) # (U, 2s+1)
        # scores = tf.reshape(scores, (U, 1+(2*self.setup.s))) # (U, 2s+1)
        
        # bestShift = tf.math.argmax(scores, axis = 1)        # (U)
        # scores = tf.gather(scores, bestShift, batch_dims=1) # (U)
        # losses = tf.gather(losses, bestShift, batch_dims=1) # (U)            
        # #print("[DEBUG] >>> U:", U)
        # #print("[DEBUG] >>> bestShift shape:", bestShift.shape)
        # #print("[DEBUG] >>> gathered scores shape:", scores.shape)
        # #print("[DEBUG] >>> gathered losses shape:", losses.shape)

        # if self.setup.s > 0:
        #     # exclude best shifts at edges
        #     shiftMask = tf.logical_not(tf.logical_or(tf.equal(bestShift, 0), tf.equal(bestShift, 2*self.setup.s))) 
        # else:
        #     # nothing to exclude
        #     shiftMask = tf.constant(True, shape=bestShift.shape)

        # #print("[DEBUG] >>> shiftMask shape:", shiftMask.shape)
        # bestShift = tf.boolean_mask(bestShift, shiftMask, axis=0) # (U*)
        # scores = tf.boolean_mask(scores, shiftMask, axis=0)
        # losses = tf.boolean_mask(losses, shiftMask, axis=0)
        # #print("[DEBUG] >>> masked bestShift shape:", bestShift.shape)
        # #print("[DEBUG] >>> masked scores shape:", scores.shape)
        # #print("[DEBUG] >>> masked losses shape:", losses.shape)
        # #print("[DEBUG] >>> P2 shape:", P2.shape)
        # P2 = tf.boolean_mask(P2, shiftMask, axis=2) 
        # #print("[DEBUG] >>> masked P2 shape:", P2.shape)        
        # P2 = tf.gather(tf.transpose(P2, [2,3,0,1]), indices=bestShift, batch_dims=1) # (U*, k, alphabet_size)
        # #print("[DEBUG] >>> gathered P2 shape:", P2.shape)
        # P2 = tf.transpose(P2, [1,2,0]) # (k, alphabet_size, U*)
        # #print("[DEBUG] >>> transposed P2 shape:", P2.shape)
        
        # return P2, scores.numpy(), losses.numpy()