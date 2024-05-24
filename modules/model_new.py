#!/usr/bin/env python
import logging
import numpy as np
import tensorflow as tf
from time import time

from . import dataset
from . import ModelDataSet
from . import ProfileFindingSetup
from . import sequtils as su



class ProfileHistory:
    """ Track the best profiles found in the last cache_size epochs. """
    def __init__(self, cache_size: int):
        assert cache_size > 0, "[ERROR] >>> cache_size must be a positive integer"
        self.cache_size = cache_size
        self.best_profile_idx = np.ndarray((cache_size,), dtype=int)
        self.best_profile_score = np.ndarray((cache_size,), dtype=float)
        self.rotating_idx = 0
        self.epoch_count = 0

    def update(self, best_profile_idx, best_profile_score):
        self.best_profile_idx[self.rotating_idx] = best_profile_idx
        self.best_profile_score[self.rotating_idx] = best_profile_score
        self.epoch_count += 1
        self.rotating_idx += 1
        if self.rotating_idx == self.cache_size:
            self.rotating_idx = 0


class TrainingStepHistory:
    """ Collect metrics during a training step. Can later be reported to the overall history """
    def __init__(self):
        self.Smin = float('inf')
        self.Smax = float('-inf')
        self.Rmin = float('inf')
        self.Rmax = float('-inf')
        self.losses = []

    def update(self, S: tf.Tensor, R: tf.Tensor, loss: float):
        self.Smin = min(self.Smin, tf.reduce_min(S).numpy())
        self.Smax = max(self.Smax, tf.reduce_max(S).numpy())
        self.Rmin = min(self.Rmin, tf.reduce_min(R).numpy())
        self.Rmax = max(self.Rmax, tf.reduce_max(R).numpy())
        self.losses.append(loss)
            


# TODO: Rewrite ProfileFindingSetup to work with ModelDataSet (should be simpler)

class SpecificProfile(tf.keras.Model):
    def __init__(self, 
                 setup: ProfileFindingSetup.ProfileFindingTrainingSetup,
                 data: ModelDataSet.ModelDataSet,
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
        self.data = data
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.setup.learning_rate)

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

    
    def _getRandomProfiles(self):
        """ Returns a random profile matrix of shape (k+(2*s), alphabet_size, U). """
        Q1 = tf.expand_dims(self.setup.data.Q, 0)
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
            if self.setup.k != 20:
                logging.warning("[model.getP] >>> phylo_t > 0 requires amino acid alphabet and k=20, " + \
                                f"not {self.setup.k}")
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
        Q1 = tf.expand_dims(self.setup.data.Q, 0)
        Q2 = tf.expand_dims(Q1, -1)
        # Limit the odds-ratio, to prevent problem with log(0).
        # Very bad matches of profiles are irrelevant anyways.
        ratio = tf.maximum(P/Q2, self.epsilon)
        R = tf.math.log(ratio)
        if tf.reduce_any(tf.math.is_nan(P)):
            logging.debug(f"[model.getR] >>> nan in P: {tf.reduce_any(tf.math.is_nan(P), axis=[0,1])} " + \
                          f"{tf.boolean_mask(P, tf.reduce_any(tf.math.is_nan(P), axis=[0,1]), axis=2)}")
            logging.debug(f"[model.getR] >>> Q: {self.setup.data.Q}")
            
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
            #print("[DEBUG] >>> nan in R")
            logging.debug("[model.getZ] >>> nan in R")
        if tf.reduce_any(tf.math.is_nan(X)):
            #print("[DEBUG] >>> nan in X")
            logging.debug("[model.getZ] >>> nan in X")
        if tf.reduce_any(tf.math.is_nan(Z)):
            #print("[DEBUG] >>> nan in Z")
            logging.debug("[model.getZ] >>> nan in Z")
        
        return Z, R


    def call(self, X, P):
        """ Returns S, R, Z; shapes are (ntiles, N, U), (k, alphabet_size, U) and (ntiles, N, f, tile_size-k+1, U). 
            Argument `P` must be _softmaxed_, don't pass the logits! """
        Z, R = self.getZ(X, P)

        S = tf.reduce_max(Z, axis=[2,3])   # shape (ntiles, N, U)
        return S, R, Z
    

    def loss(self, Z, P):
        """ Returns the score (float) and the loss per profile (shape (U)).
            Scores is the max loss over all tiles and frames, summed up for all genomes and profiles.
            Loss per profile is the softmax over all positions (tiles, frames) per genome and profile, maxed for each
               profile and summed over all genomes. """
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
        # shape of P: (k+2s, alphabet_size, U)
        L2 = tf.reduce_sum(tf.math.square(P), axis=[0,1]) # U
        L2 = tf.math.divide(L2, P.shape[0])
        L2 = tf.math.multiply(L2, self.setup.l2)
        loss_by_unit = tf.math.add(loss_by_unit, L2)      # U
        
        return score, loss_by_unit
    
    
    @tf.function()
    def train_step(self, X):
        with tf.GradientTape() as tape:
            S, R, Z = self.call(X, self.getP())
            score, loss_by_unit = self.loss(Z, self.P_logit) # TODO: what P to pass to loss? In old version, this switches (otherP is usually softmaxed, but here it's the logits)
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
        
        # TODO: take care of tracking stuff later
        # # [DEBUG] update first max_scores in tracking if desired 
        # if self.setup.trackProfiles is not None and len(self.setup.trackProfiles) > 0:
        #     assert len(self.tracking['max_score']) == 1, str(self.tracking)
        #     assert len(self.tracking['P']) == 1, str(self.tracking)
        #     self.tracking['max_score'][0] = self.max_profile_scores(self.setup.getDataset(), 
        #                                                             otherP = self.tracking['P'][0])
        # # [DEBUG]/

        # start training loop
        training_start_time = time()
        epoch_count = 0
        profileHist = ProfileHistory(self.setup.profile_plateau)
        edgecase_count = 0
        run = True
        while run:
            steps = 0
            ds_train = self.data.getDataset(repeat = True)
            trainstepHist = TrainingStepHistory()
            for batch, _ in ds_train: # shape: (batchsize, ntiles, N, f, tile_size, alphabet_size)
                for X in batch:       # shape: (ntiles, N, f, tile_size, alphabet_size)
                    assert len(X.shape) == 5, str(X.shape)
                    S, R, loss = self.train_step(X)
                    trainstepHist.update(S, R, loss.numpy())
                    
                steps += 1
                if steps >= self.setup.steps_per_epoch:
                    break
                    
            ds_eval = self.data.getDataset(withPosTracking = True)
            best_profile_idx, best_profile_mean_loss = self.get_best_profile(ds_eval)

            # TODO: take care of tracking stuff later
            # [DEBUG]
            # self.P_report_bestlosshist.append(s)
            # self.P_report_bestlosshistIdx.append(p)
            
            # # [DEBUG] track profiles
            # if self.setup.trackProfiles is not None and len(self.setup.trackProfiles) > 0:
            #     Pt = tf.gather(self.getP(), self.setup.trackProfiles, axis=2)
            #     self.tracking['epoch'].append(i+1)
            #     self.tracking['P'].append(Pt)
            #     self.tracking['max_score'].append(self.max_profile_scores(self.setup.getDataset(), otherP = Pt))
            
            # #print("epoch", i, "best profile", p.numpy(), "with score", s.numpy())
            # profileHist['idx'][profileHist['i']] = p
            # profileHist['score'][profileHist['i']] = s.numpy()
            # profileHist['i'] = 0 if profileHist['i']+1 == self.setup.profile_plateau else profileHist['i']+1
            # profileHist['c'] += 1

            # self.history['loss'].append(np.mean(Lb))
            # self.history['Rmax'].append(Rmax)
            # self.history['Rmin'].append(Rmin)
            # self.history['Smax'].append(Smax)
            # self.history['Smin'].append(Smin)
            # self.history['learning rate'].append(learning_rate)

            # check if a profile can be reported and report it

            if max_epochs is not None and profileHist.epoch_count > max_epochs:
                logging.warning("[model.train] >>> Could not find a good profile in time, " + \
                                f"force report of profile {best_profile_idx.numpy()}")
                edgecase = self.profile_cleanup(best_profile_idx)
                edgecase_count = edgecase_count+1 if edgecase else 0 # increase or reset edgecase count
                    
                # reset training
                logging.debug("[model.train] >>> Resetting training")
                profileHist = ProfileHistory(self.setup.profile_plateau)
                learning_rate = self.setup.learning_rate
                setLR(learning_rate)

            else:
                if profileHist.epoch_count >= self.setup.profile_plateau \
                                                            and all(profileHist.best_profile_idx == best_profile_idx):
                    stdev = np.std(profileHist.best_profile_score)
                    if stdev <= self.setup.profile_plateau_dev:
                        logging.info(f"[model.train] >>> epoch {epoch_count} best profile " \
                                     + f"{best_profile_idx.numpy()} with mean loss {best_profile_mean_loss.numpy()}")
                        logging.info(f"[model.train] >>> cleaning up profile {best_profile_idx.numpy()}")
                        
                        edgecase = self.profile_cleanup(best_profile_idx)
                        edgecase_count = edgecase_count+1 if edgecase else 0 # increase or reset edgecase count
                            
                        # TODO: take care of tracking stuff later
                        # [DEBUG] track profiles
                        # if len(self.tracking['masking']) > 0 and self.tracking['masking'][-1]['after_epoch'] is None:
                        #     self.tracking['masking'][-1]['after_epoch'] = i
                            
                        # reset training
                        logging.debug("[model.train] >>> Resetting training")
                        profileHist = ProfileHistory(self.setup.profile_plateau)
                        learning_rate = self.setup.learning_rate
                        setLR(learning_rate)
                    
            # log training progress in certain steps
            if verbose and (epoch_count % (verbose_freq) == 0 \
                            or (self.setup.n_best_profiles is None and epoch_count == self.setup.epochs-1)):
                tnow = time()
                _, R, Z = self.call(X, self.getP())
                _, loss_by_unit = self.loss(Z, self.P_logit)
                logging.info(f"[model.train] >>> epoch {epoch_count} best profile {best_profile_idx.numpy()} " \
                             + f"with mean loss {best_profile_mean_loss.numpy()}")
                logging.info(f"[model.train] >>> epoch {epoch_count:>5} profile loss sum " + \
                             f"= {tf.reduce_sum(loss_by_unit).numpy():.4f}" + \
                             " max R: {:.3f}".format(tf.reduce_max(R).numpy()) + \
                             " min R: {:.3f}".format((tf.reduce_min(R).numpy())) + \
                             " time: {:.2f}".format(tnow-training_start_time)) 

            # check if learning rate should decrease
            if len(self.history['loss']) > self.setup.lr_patience:
                lastmin = self.history['loss'][-(self.setup.lr_patience+1)] # loss before the last lr_patience epochs
                if not any([l < lastmin for l in self.history['loss'][-self.setup.lr_patience:]]):
                    logging.info("[model.train_reporting.reduceLR] >>> Loss did not decrease for " + \
                                 f"{self.setup.lr_patience} epochs, reducing learning rate from {learning_rate} to " + \
                                 f"{self.setup.lr_factor*learning_rate}")
                    learning_rate *= self.setup.lr_factor
                    setLR(learning_rate)

            # determine if training should continue
            epoch_count += 1
            if self.setup.n_best_profiles is not None:
                run = (len(self.P_report) < self.setup.n_best_profiles) # TODO: tracking stuff
                if edgecase_count > 10:
                    logging.warning("[model.train_reporting] >>> Training seems to be stuck in edge cases, aborting")
                    run = False
            else:
                run = (epoch_count < self.setup.epochs)
                



    def get_best_profile(self, ds):
        """ Return index of the profile that has the lowest loss """
        losses = []
        for batch in ds:
            X = batch[0]        # (B, tilePerX, N, f, tileSize, 21)
            posTrack = batch[1] # (B, tilePerX, N, f, 4)
            assert len(X.shape) == 6, str(X.shape)
            assert posTrack.shape != (1, 0), str(posTrack.shape)+" -- use batch dataset with position tracking!"
            assert X.shape[0:4] == posTrack.shape[0:4], f"{X.shape=} != {posTrack.shape=}"
            for b in range(X.shape[0]): # iterate samples in batch
                _, _, Z = self.call(X[b], self.getP())       # Z: (ntiles, N, f, tile_size-k+1, U)
                _, loss_by_unit = self.loss(Z, self.P_logit) # (U)

                # (tilePerX, N, f) -> -1 if tile was exhausted -> False if exhausted -> 1 for valid tile, else 0
                W = tf.cast(posTrack[b,:,:,:,0] != -1, tf.float32) # binary mask for valid tiles
                W = tf.multiply(tf.reduce_sum(W), tf.ones((self.setup.U), dtype=tf.float32)) # weight for the means
                losses.append(tf.multiply(loss_by_unit, W).numpy()) # store weighted losses

                if tf.reduce_any( tf.math.is_nan(Z) ):
                    logging.debug("[model.get_best_profile] >>> nan in Z")
                    logging.debug(f"[model.get_best_profile] >>> W: {W}")
                    
        B = tf.reduce_mean(losses, axis=0) # get overall lowest mean loss per profile
        
        # TODO: take care of tracking stuff later
        # # [DEBUG] track profile losses over time
        # if self.P_report_plosshist is None:
        #     self.P_report_plosshist = tf.reshape(B, (1,-1))
        # else:
        #     self.P_report_plosshist =  tf.concat([self.P_report_plosshist, tf.reshape(B, (1,-1))], axis=0)
        
        return tf.argmin(B), tf.reduce_min(B) # profile index, mean loss