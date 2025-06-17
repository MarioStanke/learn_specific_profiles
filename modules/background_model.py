""" Module for implementations of higher order background models. """

from Bio import SeqIO
import numpy as np
from pathlib import Path
from time import time

from . import ModelDataSet

# human intergenic regions dinucleotide frequencies from Augustus
HG_INTERGENIC_DINT_FREQ = {
    'AA': 0.0916,
    'AC': 0.0506,
    'AG': 0.0717,
    'AT': 0.0693,
    'CA': 0.0728,
    'CC': 0.0586,
    'CG': 0.0137,
    'CT': 0.0717,
    'GA': 0.0596,
    'GC': 0.048,
    'GG': 0.0586,
    'GT': 0.0506,
    'TA': 0.0592,
    'TC': 0.0596,
    'TG': 0.0728,
    'TT': 0.0916
}


def _get_augustus_model():
    """
    Get the Augustus model for human intergenic regions. Returns a 4x4 matrix
    representing the model, a dict with dinucleotide frequencies and the alphabet used (ACGT).
    The matrix is constructed such that each row sums to 1, and the values
    are non-negative. The model is based on the dinucleotide frequencies
    from Augustus for human intergenic regions.
    """
    assert len(HG_INTERGENIC_DINT_FREQ) == 16, "Expected 16 dinucleotide frequencies"
    assert sum(HG_INTERGENIC_DINT_FREQ.values()) == 1.0, "Expected sum of dinucleotide frequencies to be 1.0"
    model = np.zeros((4, 4))
    alphabet = "ACGT"
    for i, a in enumerate(alphabet):
        s = sum(HG_INTERGENIC_DINT_FREQ[a + b] for b in alphabet)
        for j, b in enumerate(alphabet):
            model[i, j] = HG_INTERGENIC_DINT_FREQ[a + b] / s

        assert abs(sum(model[i, :]) - 1.0) < 1e-6, f"Expected sum of row {i} to be 1.0, got {sum(model[i, :])}"
        assert all(model[i, :] >= 0), f"Expected all values in row {i} to be non-negative, got {model[i, :]}"

    return model, HG_INTERGENIC_DINT_FREQ.copy(), alphabet


def _get_uniform_model(order: int):
    """
    Get a uniform background model for a given order. The model is a {4}^(order+1) matrix
    representing the uniform distribution of nucleotides. The alphabet used is ACGT.
    Returns a tuple of the model (np.ndarray) and the frequencies of the nucleotides (dict) 
    and the alphabet used (str 'ACGT').
    """
    assert order >= 0, "Order must be greater than or equal to 0"
    alphabet = "ACGT"
    size = len(alphabet)
    model = np.full((size,)*(1+order), 1.0 / size)
    if order == 0:
        freqs = {c: 1.0 / size for c in alphabet}
    else:
        freqs = {}
        for i in range(len(alphabet)**(order + 1)):
            kmer = "".join([alphabet[i // (len(alphabet)**j) % len(alphabet)] for j in range((order + 1))])
            freqs[kmer] = 1.0 / (len(alphabet)**(order + 1))
    return model, freqs, alphabet


def get_background_model(order: int, model_type: str = "uniform", src: Path = None):
    """
    Get a background model for a given order and model type. The model can be either
    'uniform', 'data' or 'augustus'. The uniform model is a uniform distribution of nucleotides,
    while the data model is based on the dinucleotide frequencies from a given file.
    The Augustus model is based on the dinucleotide frequencies from Augustus for
    human intergenic regions and is restricted to order 1.
    The function returns a tuple of the model (np.ndarray) and the frequencies of the nucleotides (dict)
    and the alphabet used (str 'ACGT').
    """
    assert order >= 0, "Order must be greater than or equal to 0"
    assert model_type in ["uniform", "data", "augustus"], "Model type must be either 'uniform', 'data' or 'augustus'"

    if model_type == "uniform":
        if src is not None:
            print(f"[Warning] >>> Source file {src} is ignored for uniform model")
        return _get_uniform_model(order)
    elif model_type == "augustus":
        assert order == 1, "Order must be 1 for Augustus model"
        if src is not None:
            print(f"[Warning] >>> Source file {src} is ignored for uniform model")
        return _get_augustus_model()
    elif model_type == "data":
        alphabet = "ACGT"
        assert src is not None, "Source file must be provided for data model"
        assert src.exists(), f"Source file {src} does not exist"
        records = [r for r in SeqIO.parse(src, "fasta")]
        assert len(records) > 0, f"Source file {src} is empty or contains no records"
        assert all(len(r.seq) > 0 for r in records), f"Source file {src} contains empty sequences"
        freqs = {}
        k = order + 1
        # it's possible that not all k-mers are present in the sequences, so we need to
        # initialize the frequencies to 0
        if order == 0:
            freqs = {c: 0 for c in alphabet}
        else:
            for i in range(len(alphabet)**k):
                kmer = "".join([alphabet[i // (len(alphabet)**j) % len(alphabet)] for j in range(k)])
                freqs[kmer] = 0
        # count the frequencies of k-mers in the sequences
        for record in records:
            seq = str(record.seq)
            for i in range(len(seq) - k):
                kmer = seq[i:i + k]
                # ignore ambiguous or softmasked nucleotides
                if all(c in alphabet for c in kmer):    
                    assert kmer in freqs, f"Expected k-mer {kmer} to be in frequencies dictionary"
                    freqs[kmer] += 1
        # normalize frequencies
        total = sum(freqs.values())
        if total == 0:
            print(f"[Warning] >>> No {k=}-mers found in sequences, using uniform model")
            return _get_uniform_model(order)
        assert total > 0, f"Expected total frequency to be greater than 0, got {total}"
        # normalize frequencies to sum to 1
        for kmer in freqs:
            freqs[kmer] /= total
        # create a matrix of shape {4}^(order+1)
        size = len(alphabet)
        model = np.zeros((size,)*k)
        # print(f"[DEBUG] >>> {order=}, {k=}, {size=}, {model.shape=}")
        if order == 0:
            # 0-mers, i.e. the frequencies of the nucleotides
            for i in range(size):
                model[i] = freqs[alphabet[i]]
        else:
            for i in range(len(alphabet)**order):
                # k-1-mers, i.e. the first #order nucleotides -> add the last nucleotide later to handle missing k-mers
                o_nt = "".join([alphabet[i // (len(alphabet)**j) % len(alphabet)] for j in range(order)])
                o_idcs = tuple([alphabet.index(c) for c in o_nt])
                s = sum(freqs[o_nt + alphabet[i]] for i in range(size))
                # print(f"[DEBUG] >>> {o_nt=}, {o_idcs=}, {s=}")
                for i in range(size):
                    kmer = o_nt + alphabet[i]
                    idcs = o_idcs + (i,)
                    if s == 0:
                        model[idcs] = 1/size # k-mer was not found in the sequences, use uniform distribution
                    else:
                        model[idcs] = freqs[kmer] / s
                # print(f"[DEBUG] >>> {o_idcs=}, {model=}, {model[o_idcs][:]=}, {model[0, :]=}, {sum(model[o_idcs, :])=}")
                assert abs(sum(model[o_idcs][:]) - 1.0) < 1e-6, \
                    f"Expected sum of row {o_idcs} to be 1.0, got {sum(model[o_idcs][:])}"
                assert all(model[o_idcs][:] >= 0), \
                    f"Expected all values in row {o_idcs} to be non-negative, got {model[o_idcs][:]}"
        
        return model, freqs, alphabet



# ======================================================================================================================

import logging
import tensorflow as tf

class TrainedQ(tf.keras.Model): # type: ignore
    def __init__(self, 
                 num_models: int = 2, # K
                 rand_seed: int = None, **kwargs): # type: ignore
        """
        Set up model and most metaparamters
            Parameters:
                # setup: ProfileFindingTrainingSetup object containing metaparameters and initial profiles
                rand_seed (int): optional set a seed for tensorflow's rng
        """
        super().__init__(**kwargs)

        self.opt = tf.keras.optimizers.Adam(learning_rate=float(2))

        # setting random seeds if desired
        if rand_seed is not None:
            logging.debug(f"[model.__init__] >>> setting tf global seed to {rand_seed}")
            tf.random.set_seed(rand_seed)

        self.nprng = np.random.default_rng(rand_seed) # if rand_seed is None, unpredictable entropy is pulled from OS

        # === DRAFT ===

        self.num_models = num_models
        self.order = 0 # order of the models, i.e. k-mer-length - 1 # FOR NOW: only allow 0, higher oder needs carification: how to use it in model.py?
        self.alphabet = "ACGT"

        self.Q_logit = tf.Variable(tf.random.uniform(shape=(len(self.alphabet)**(self.order+1), self.num_models),
                                                     minval=-0.1, maxval=0.1, dtype=tf.float32), 
                                   trainable=True, name="Q_logit") # shape: (alphabet_size**(order+1), K)
        self.m_logit = tf.Variable(tf.random.uniform(shape=(self.num_models,),
                                                     minval=-0.1, maxval=0.1, dtype=tf.float32), 
                                    trainable=True, name="m_logit") # shape: (K,)
        logging.debug(f"[background_model.__init__] >>> Q_logit shape: {self.Q_logit.shape}, m_logit shape: {self.m_logit.shape}")


    def getQ(self):
        """ Returns softmaxed Q (alphabet_size**(order+1), K). """
        Q = tf.nn.softmax(self.Q_logit, axis=0, name="Q")
        return Q


    def getM(self):
        """ Returns softmaxed m (K,). """
        m = tf.nn.softmax(self.m_logit, axis=0, name="m")
        return m


    def call(self, X):
        """ Input X is a tensor of shape (batch_size, ntiles, N, f, tile_size, alphabet_size) """
        Q = self.getQ() # shape: (alphabet_size**(order+1), K)
        m = self.getM() # shape: (K,)

        batch_dims = len(X.shape) - 2 # number of batch dimensions, e.g. 4 for (batch_size, ntiles, N, f, tile_size, alphabet_size)

        Qt = tf.transpose(Q, [1, 0]) # shape: (K, alphabet_size**(order+1))
        Qr = tf.reshape(Qt, ((1,)*batch_dims)+(1,)+Qt.shape) # shape: (1,          1,      1, 1,         1, K, alphabet_size**(order+1))
        X1 = tf.expand_dims(X, -2) #                           shape: (batch_size, ntiles, N, f, tile_size, 1, alphabet_size)
        C = tf.multiply(X1, Qr) # shape: (batch_size, ntiles, N, f, tile_size, K, alphabet_size**(order+1))
        C1 = tf.reduce_sum(C, axis=-1) # shape: (batch_size, ntiles, N, f, tile_size, K), sum over alphabet_size dimension
        D = tf.math.log(tf.clip_by_value(C1, 1e-10, tf.reduce_max(C1))) # shape: (batch_size, ntiles, N, f, tile_size, K), log to avoid numerical issues
        D1 = tf.reduce_sum(D, axis=-2) # shape: (batch_size, ntiles, N, f, K), sum over tile_size dimension
        M = tf.reduce_max(D1, axis=-1) # shape: (batch_size, ntiles, N, f), max over K dimension
        D2 = D1 - tf.expand_dims(M, -1) # shape: (batch_size, ntiles, N, f, K), subtract max to avoid numerical issues
        D3 = tf.multiply(tf.exp(D2), m) # shape: (batch_size, ntiles, N, f, K), multiply by model weights
        D4 = tf.reduce_sum(D3, axis=-1) # shape: (batch_size, ntiles, N, f), sum over K dimension
        D5 = tf.math.log(tf.clip_by_value(D4, 1e-10, tf.reduce_max(D4))) # shape: (batch_size, ntiles, N, f), log to avoid numerical issues
        S = tf.add(M, D5) # shape: (batch_size, ntiles, N, f), add max back to get the final score
        # logging.debug(f"[background_model.call] >>> Q:\n{Qt}\n\nX:\n{X}\n\nC:\n{C}\n\nC1:\n{C1}\n\nD:\n{D}\n\nD1:\n{D1}\n\nM:\n{M}\n\nD2\n{D2}\n\nm:\n{m}\n\nD3\n{D3}\n\nD4\n{D4}\n\nD5\n{D5}\n\nS:\n{S}")
        # logging.debug(f"[background_model.call] >>> P:\n{tf.reduce_sum(S)}\n\nloss:\n{-tf.math.log(tf.reduce_sum(S))}") # preview on loss

        return S

        # TODO: underflow loesen

        # ==============================================================================================================

        Q1 = tf.expand_dims(Q, 0) # shape: (1, alphabet_size**(order+1), K)
        Q2 = tf.expand_dims(Q1, 2) # shape: (1, alphabet_size**(order+1), 1, K)
        X1 = tf.expand_dims(X, -1) # shape: (batch_size, ntiles, N, f, tile_size, alphabet_size, 1)
        # logging.debug(f"[background_model.call] >>> Q2 shape: {Q2.shape}, m shape: {m.shape}")
        # conv2d: input shape  (batch_shape (batch_size, ntiles, N, f), in_heigth (tile_size), in_width (alphabet_size),     in_channels (1))
        #         filter shape                                         (filter_height (1),     filter_width (alphabet_size), in_channels (1), out_channels (K))
        #         output shape (batch_shape (batch_size, ntiles, N, f), out_height, out_width, out_channels (K))
        C = tf.nn.conv2d(X1, Q2, strides=1, padding='VALID', data_format="NHWC", name="C") # shape: batch_size, ntiles, N, f, out_height, out_width (1), K)
        # logging.debug(f"[background_model.call] >>> C shape: {C.shape}")
        C1 = tf.squeeze(C, axis=-2) # shape: (batch_size, ntiles, N, f, out_height, K), remove the out_width dimension
        # logging.debug(f"[background_model.call] >>> C1 shape: {C1.shape}")
        C2 = tf.reduce_prod(C1, axis=-2) # shape: (batch_size, ntiles, N, f, K), product over out_height dimension
        # logging.debug(f"[background_model.call] >>> C2 shape: {C2.shape}")
        # multiply last dimension with m, i.e. the model weights
        C3 = tf.multiply(C2, m) # shape: (batch_size, ntiles, N, f, K)
        # logging.debug(f"[background_model.call] >>> C3 shape: {C3.shape}")
        # sum over the last dimension, i.e. the models
        S = tf.reduce_sum(C3, axis=-1) # shape: (batch_size, ntiles, N, f)
        # logging.debug(f"[background_model.call] >>> S shape: {S.shape}")

        return S # shape: (batch_size, ntiles, N, f)


    def lossfun(self, S):
        P = tf.reduce_sum(S)
        #return -tf.math.log(tf.clip_by_value(P, 1e-10, tf.reduce_max(P))) # add small value to avoid log(0)
        return -P # P already in log space


    @tf.function()
    def train_step(self, X):
        # TODO: müsste hier nicht erst einmal ein ganzer Batch trainiert werden, bevor Gradient berechnet wird?
        with tf.GradientTape() as tape:
            S = self.call(X)
            loss = self.lossfun(S)
            
        grad = tape.gradient(loss, [self.Q_logit, self.m_logit])
        #logging.debug(f"[background_model.train_step] >>> S:\n{S}\n\nloss:\n{loss}\n\ngrad:\n{grad}")
        # logging.debug(f"[background_model.train_step] >>> grad shape: {[g.shape for g in grad]}")
        self.opt.apply(grad, [self.Q_logit, self.m_logit])
        
        return S, loss#, grad


    def train(self, data: ModelDataSet.ModelDataSet, lr=2, epochs=50,
              verbose=True, verbose_freq=10):
        def setLR(learning_rate):
            logging.debug(f"[model.train.setLR] >>> Setting learning rate to {learning_rate}")
            self.opt.learning_rate.assign(learning_rate)

        max_epochs = epochs
        steps_per_epoch = data.getStepsPerEpoch() # use the steps_per_epoch from the dataset, this should be accurate
        learning_rate = lr # gets altered during training
        setLR(learning_rate) # reset learning rate to initial value for safety

        # start training loop
        training_start_time = time()
        epoch_count = 0
        losses = []
        lr_reduction_cooldown = 0
        run = True
        while run:
            # run an epoch
            steps = 0
            ds_train = data.getDataset(repeat = True)
            _bshape = None
            for batch, _ in ds_train: # shape: (batchsize, ntiles, N, f, tile_size, alphabet_size)
                # for X in batch:       # shape: (ntiles, N, f, tile_size, alphabet_size)
                #     assert len(X.shape) == 5, str(X.shape)
                #     self.train_step(X)
                assert len(batch.shape) == 6, str(batch.shape)
                _bshape = batch.shape
                self.train_step(batch)

                steps += 1
                if steps >= steps_per_epoch:
                    break
                    
            lossls = []
            ds_loss = data.getDataset(repeat = False)
            for batch, _ in ds_loss: # shape: (batchsize, ntiles, N, f, tile_size, alphabet_size)
                # for X in batch:       # shape: (ntiles, N, f, tile_size, alphabet_size)
                #     assert len(X.shape) == 5, str(X.shape)
                #     S = self.call(X)
                #     lossls.append(self.lossfun(S))
                assert len(batch.shape) == 6, str(batch.shape)
                S = self.call(batch)
                lossls.append(self.lossfun(S))

            losses.append(tf.reduce_mean(lossls).numpy())

            # log training progress in certain steps
            if verbose and (epoch_count % (verbose_freq) == 0 or epoch_count == max_epochs-1):
                tnow = time()
                logging.info(f"[background_model.train] >>> epoch {epoch_count:>5} mean loss = {losses[-1]:<.4f}," + \
                             f" time: {tnow-training_start_time:.2f}s") 
                logging.debug(f"[background_model.train] >>> batch shape: {_bshape}")

            # check if learning rate should decrease
            lr_reduction_cooldown -= 1
            if lr_reduction_cooldown <= 0 and len(losses) > 10:
                lastmin = losses[-11] # loss before the last lr_patience epochs
                if not any([l < lastmin for l in losses[-10:]]):
                    logging.info("[background_model.train] >>> Loss did not decrease for " + \
                                 f"{10} epochs, reducing learning rate from {learning_rate} to " + \
                                 f"{0.75*learning_rate}")
                    learning_rate *= 0.75
                    setLR(learning_rate)
                    lr_reduction_cooldown = 10 # do not immediately reduce again after a reduction

            # determine if training should continue
            epoch_count += 1
            run = (epoch_count < max_epochs)

        return losses # return the mean losses for each epoch

    # # def convertX(self, X):
    # #     """ Converts the data input X of shape (ntiles, N, f, tile_size, alphabet_size) to a tensor of shape 
    # #         (ntiles, N, f, tile_size-order, alphabet_size**(order+1), 1), i.e. a series of overlapping k-mers (
    # #         where k = order+1), with an additional input channel dimension. """
    # #     assert len(X.shape) == 5, str(X.shape)
    # #     ntiles, N, f, tile_size, alphabet_size = X.shape
    # #     # the alphabet_size is the number of different characters in the alphabet, e.g. 4 for DNA
    # #     # the respective dimension contains one-hot encoded characters, i.e. 1 for the character and 0 for all others
    # #     # -> the desired output shape is (ntiles, N, f, tile_size-order, alphabet_size**(order+1), 1), where the single
    # #     #    character encoding is replaced by a k-mer encoding of all overlapping k-mers in the input
    # #     tile_size_k = tile_size - self.order # k = order + 1
    # #     # TODO: either find a smart way to implement this here or implement it during data set creation, only needed when higher orders are allowed


    # def lossfun(self, Z, P_logit): # TODO: rewrite for this use case! vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    #     """ Returns the score (float) and the loss per profile (shape (U)).
    #         Scores is the max loss over all tiles and frames, summed up for all genomes and profiles.
    #         Loss per profile is the softmax over all positions (tiles, frames) per genome and profile, maxed for each
    #            profile and summed over all genomes. 
    #         Pass P_logit _instead of softmaxed P_, as the L2 regularization is weaker with value ranges close to 0. For
    #            KLD regularization, P_logit is softmaxed when calculating the regularization term. """
    #     # shape of Z: ntiles x N x f x tile_size-k+1 x U 
    #     S = tf.reduce_max(Z, axis=[0,2,3]) # N x U
    #     score = tf.reduce_sum(S)
        
    #     Z = tf.transpose(Z, [1,4,0,2,3]) # shape N x U x ntiles x f x tile_size-k+1
    #     Z = tf.reshape(Z, [Z.shape[0], Z.shape[1], -1]) # shape N x U x -1
    #     Zsm = tf.nn.softmax(self.setup.gamma*Z, axis=-1) # softmax for each profile in each genome 
    #     # logging.debug(f"[model.lossfun] >>> \nZ: {Z},\nZsm: {Zsm}")
    #     Z = tf.math.multiply(Z, Zsm)
    #     # logging.debug(f"[model.lossfun] >>> \nZ: {Z}")

    #     Z = tf.maximum(Z, 0)
    #     # logging.debug(f"[model.lossfun] >>> \nZ (2): {Z}")
        
    #     loss_by_unit = -tf.math.reduce_max(Z, axis=-1) # best isolated match for each profile in each genome (N x U)
    #     # logging.debug(f"[model.lossfun] >>> \nloss_by_unit: {loss_by_unit}\n  mean: {tf.reduce_mean(loss_by_unit)}, max: {tf.reduce_max(loss_by_unit)}")
        
    #     def mellowmax(x, a):
    #         """ See https://en.wikipedia.org/wiki/Smooth_maximum#Mellowmax
    #           mm(x, a) = 1/a log( sum(exp(x*a)) / n ) --> ( log(sum(exp(x*a))) - log(n) ) / a 
    #           (according to https://stackoverflow.com/a/76608729 to avoid possible overflow)

    #         x should have shape (N, U)
    #         """
    #         # logging.debug(f"[model.lossfun] >>> mellowmax called with {x.shape=} and {a=}")
    #         n = x.shape[0] # N
    #         x = tf.math.multiply(x, a) # (N, U)
    #         lse = tf.math.log(tf.reduce_sum(tf.exp(x), axis=0)) # (U)
    #         return tf.math.divide(tf.math.subtract(lse, tf.math.log(tf.cast(n, x.dtype))), a)

    #     # logging.debug(f"[model.lossfun] >>> \nmellowmax(loss,  0.01): {mellowmax(loss_by_unit, 0.01)}")
    #     # logging.debug(f"[model.lossfun] >>> \nmellowmax(loss,  0.1): {mellowmax(loss_by_unit, 0.1)}")
    #     # logging.debug(f"[model.lossfun] >>> \nmellowmax(loss,  1  ): {mellowmax(loss_by_unit, 1)}")
    #     # logging.debug(f"[model.lossfun] >>> \nmellowmax(loss, 10  ): {mellowmax(loss_by_unit, 10)}")

    #     loss_by_unit = mellowmax(loss_by_unit, self.setup.mellowmax_alpha)
    #     # loss_by_unit = tf.math.reduce_sum(loss_by_unit, axis=0) # best isolated match of all genomes (U,)
    #     # logging.debug(f"[model.lossfun] >>> \nloss_by_unit (2): {loss_by_unit}")
            
    #     if self.setup.l2 != 0:
    #         # L2 regularization
    #         # use P_logit here instead of softmaxed P, as the L2 regularization is weaker with value ranges close to 0
    #         # shape of P_logit: (k+2s, alphabet_size, U)
    #         L2 = tf.reduce_sum(tf.math.square(P_logit), axis=[0,1]) # U
    #         L2 = tf.math.divide(L2, P_logit.shape[0])
    #         L2 = tf.math.multiply(L2, self.setup.l2)
    #         loss_by_unit = tf.math.add(loss_by_unit, L2)      # U

    #     if self.setup.kld != 0:
    #         # Kullback-Leibler divergence regularization 
    #         #   (adjusted implementation of https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLD)
    #         # Use softmaxed P instead of P_logit, as the KLD is not defined for logits
    #         Q = tf.clip_by_value(self.data.Q, self.epsilon, 1.0) # avoid numerical issues (log(0), division by zero)
    #         Q1 = tf.repeat(tf.expand_dims(Q, axis=0), P_logit.shape[0], axis=0)    # shape: (k+2s, alphabet_size)
    #         Q2 = tf.repeat(tf.expand_dims(Q1, axis=-1), P_logit.shape[2], axis=-1) # shape: (k+2s, alphabet_size, U)
    #         P = tf.clip_by_value(tf.nn.softmax(P_logit, axis=1), self.epsilon, 1.0) # avoid numerical issues
    #         KLD = tf.math.multiply(P, tf.math.log( tf.divide(P, Q2) )) # shape: (k+2s, alphabet_size, U)
    #         KLD = tf.reduce_sum(KLD, axis=[0,1]) # U
    #         KLD = tf.math.multiply(KLD, self.setup.kld)
    #         loss_by_unit = tf.math.add(loss_by_unit, KLD) # U
            
    #     return score, loss_by_unit
    
    

    # @tf.function()
    # def train_step(self, X):
    #     with tf.GradientTape() as tape:
    #         S, R, Z = self.call(X, self.getP())
    #         score, loss_by_unit = self.lossfun(Z, self.P_logit)
    #         # Mario's loss
    #         #loss = -score
    #         loss = tf.reduce_sum(loss_by_unit)
            
    #     grad = tape.gradient(loss, self.P_logit)
    #     self.opt.apply_gradients([(grad, self.P_logit)])
        
    #     return S, R, loss
    


    # def train(self, data:  verbose=True, verbose_freq=100):
    #     """ setup.epochs is the number of epochs to train if n_best_profiles is None, otherwise it's the max number
    #           of epochs to wait before a forced profile report """
        
    #     def setLR(learning_rate):
    #         logging.debug(f"[model.train.setLR] >>> Setting learning rate to {learning_rate}")
    #         self.opt.learning_rate.assign(learning_rate)

    #     max_epochs = self.setup.epochs
    #     learning_rate = self.setup.learning_rate # gets altered during training
    #     setLR(learning_rate) # reset learning rate to initial value for safety

    #     # start training loop
    #     training_start_time = time()
    #     epoch_count = 0
    #     run = True
    #     while run:
    #         # run an epoch
    #         steps = 0
    #         ds_train = self.data.getDataset(repeat = True)
    #         epochHist = EpochHistory()
    #         for batch, _ in ds_train: # shape: (batchsize, ntiles, N, f, tile_size, alphabet_size) # type: ignore
    #             for X in batch:       # shape: (ntiles, N, f, tile_size, alphabet_size)
    #                 assert len(X.shape) == 5, str(X.shape)
    #                 S, R, loss = self.train_step(X) # type: ignore
    #                 epochHist.update(S, R, loss.numpy())
                    
    #             steps += 1
    #             if steps >= self.setup.steps_per_epoch:
    #                 break
                    
    #         mean_losses = self.get_mean_losses(self.data.getDataset(withPosTracking = True), 
    #                                            self.getP(), self.P_logit) # (U)
    #         best_profile = tf.argmin(mean_losses).numpy()
    #         best_profile_mean_loss = tf.reduce_min(mean_losses).numpy()
            
    #         # write history and tracking
    #         profilePerfCache.update(best_profile, best_profile_mean_loss)
    #         self.history.update(epochHist, learning_rate, mean_losses.numpy())
    #         if len(self.profile_tracking.tracking_ids) > 0:
    #             Pt = tf.gather(self.getP(), self.profile_tracking.tracking_ids, axis=2)
    #             scores = tf.reduce_max( self.get_profile_scores(self.data.getDataset(), P = Pt), axis=1 ).numpy()
    #             losses = tf.gather(mean_losses, self.profile_tracking.tracking_ids, axis=0).numpy()
    #             # sites, site_scores = self.get_profile_match_sites(self.data.getDataset(withPosTracking = True), Pt, 
    #             #                                                   self.setup.match_score_factor * scores)
    #             # self.profile_tracking.addEpoch(epoch_count, Pt.numpy(), scores, losses, 
    #             #                                sites.numpy(), site_scores.numpy()) # type: ignore
    #             # vvv do not track sites, possibly responsible for OOM
    #             self.profile_tracking.addEpoch(epoch_count, Pt.numpy(), scores, losses)

    #         # check if a profile can be reported and report it
    #         if profilePerfCache.epoch_count >= self.setup.profile_plateau \
    #                                                               and all(profilePerfCache.profile_idx == best_profile):
    #             stdev = np.std(profilePerfCache.profile_score)
    #             if stdev <= self.setup.profile_plateau_dev:
    #                 logging.info(f"[model.train] >>> epoch {epoch_count} best profile " \
    #                                 + f"{best_profile} with mean loss {best_profile_mean_loss}")
    #                 logging.info(f"[model.train] >>> cleaning up profile {best_profile}")
                    
    #                 edgecase = self.profile_cleanup(best_profile, epoch_count)
    #                 edgecase_count = edgecase_count+1 if edgecase else 0 # increase or reset edgecase count
                        
    #                 # reset training
    #                 logging.debug("[model.train] >>> Resetting training")
    #                 profilePerfCache = ProfilePerformanceCache(self.setup.profile_plateau)
    #                 learning_rate = self.setup.learning_rate
    #                 setLR(learning_rate)

    #         # if no profile has been found for too long, force report the current best
    #         if profilePerfCache.epoch_count > max_epochs:
    #             logging.warning("[model.train] >>> Could not find a good profile in time, " + \
    #                             f"force report of profile {best_profile}")
    #             edgecase = self.profile_cleanup(best_profile, epoch_count)
    #             edgecase_count = edgecase_count+1 if edgecase else 0 # increase or reset edgecase count
                    
    #             # reset training
    #             logging.debug("[model.train] >>> Resetting training")
    #             profilePerfCache = ProfilePerformanceCache(self.setup.profile_plateau)
    #             learning_rate = self.setup.learning_rate
    #             setLR(learning_rate)
                
    #         # log training progress in certain steps
    #         if verbose and (epoch_count % (verbose_freq) == 0 \
    #                         or (self.setup.n_best_profiles is None and epoch_count == self.setup.epochs-1)):
    #             tnow = time()
    #             losses, _ = self.get_profile_losses(self.data.getDataset(withPosTracking=True), 
    #                                                 self.getP(), self.P_logit)
    #             logging.info(f"[model.train] >>> epoch {epoch_count} best profile {best_profile} " \
    #                          + f"with mean loss {best_profile_mean_loss}")
    #             logging.info(f"[model.train] >>> epoch {epoch_count:>5} sum of profile tile losses " + \
    #                          f"= {tf.reduce_sum(losses).numpy():.4f}," + \
    #                          f" max R: {epochHist.Rmax:.3f}, min R: {epochHist.Rmin:.3f}," + \
    #                          f" time: {tnow-training_start_time:.2f}s") 

    #         # check if learning rate should decrease
    #         lr_reduction_cooldown -= 1
    #         if lr_reduction_cooldown <= 0 and len(self.history.loss) > self.setup.lr_patience:
    #             lastmin = self.history.loss[-(self.setup.lr_patience+1)] # loss before the last lr_patience epochs
    #             if not any([l < lastmin for l in self.history.loss[-self.setup.lr_patience:]]):
    #                 logging.info("[model.train_reporting.reduceLR] >>> Loss did not decrease for " + \
    #                              f"{self.setup.lr_patience} epochs, reducing learning rate from {learning_rate} to " + \
    #                              f"{self.setup.lr_factor*learning_rate}")
    #                 learning_rate *= self.setup.lr_factor
    #                 setLR(learning_rate)
    #                 lr_reduction_cooldown = self.setup.lr_patience # do not immediately reduce again after a reduction

    #         # determine if training should continue
    #         epoch_count += 1
    #         if self.setup.n_best_profiles is not None:
    #             run = (len(self.profile_report) < self.setup.n_best_profiles)
    #             if edgecase_count > 10:
    #                 logging.warning("[model.train_reporting] >>> Training seems to be stuck in edge cases, aborting")
    #                 run = False
    #         else:
    #             run = (epoch_count < max_epochs)
                


    # def get_profile_losses(self, ds, P, P_logit):
    #     """ Argument `P` must be _softmaxed_, P_logit is the logits of P (i.e. before softmaxing)!
    #         Returns a tensor of losses for each tile of shape (U, x) where x is number_of_batches * batch_size,
    #         and a tensor of weights of the same shape (U, x): In each batch <x>, the weight for all profiles <U> is the
    #         same; the weights are 1 if all tiles in all genomes and frames are valid, or smaller if some where 
    #         exhausted. The weight tensor can be used to compute a weighted mean loss per profile. """
    #     U = P.shape[-1]
    #     losses = tf.zeros([U, 0], dtype=tf.float32) # shape (U, 0)
    #     weights = tf.zeros([U, 0], dtype=tf.float32) # shape (U, 0)
    #     for batch in ds:
    #         X = batch[0]        # (B, tilePerX, N, f, tileSize, 21)
    #         posTrack = batch[1] # (B, tilePerX, N, f, 4)
    #         assert len(X.shape) == 6, str(X.shape)
    #         assert posTrack.shape != (1, 0), str(posTrack.shape)+" -- use batch dataset with position tracking!"
    #         assert X.shape[0:4] == posTrack.shape[0:4], f"{X.shape=} != {posTrack.shape=}"
    #         ntiles = np.prod(posTrack.shape[1:4]) # tilesPerX * N * f
    #         for b in range(X.shape[0]): # iterate samples in batch
    #             _, _, Z = self.call(X[b], P)               # Z: (ntiles, N, f, tile_size-k+1, U)
    #             _, loss_by_unit = self.lossfun(Z, P_logit) # (U)

    #             # (tilePerX, N, f) -> -1 if tile was exhausted -> False if exhausted -> 1 for valid tile, else 0
    #             W = tf.cast(posTrack[b,:,:,:,0] != -1, tf.float32) # binary mask for valid tiles, (tilePerX, N, f)
    #             W = tf.reduce_sum(W) / ntiles # weight for the tile, scalar
    #             W = tf.broadcast_to(W, (U, 1)) # weight for the tile, (U, 1)
                
    #             losses = tf.concat([losses, tf.expand_dims(loss_by_unit, -1)], axis=1)
    #             weights = tf.concat([weights, W], axis=1)
            
    #             if tf.reduce_any( tf.math.is_nan(Z) ):
    #                 logging.debug("[model.get_profile_losses] >>> nan in Z")
    #                 logging.debug(f"[model.get_profile_losses] >>> W: {W}")

    #     return losses, weights



    # def get_mean_losses(self, ds, P, P_logit):
    #     """ A wrapper around get_profile_losses that returns the weighted mean loss per profile. 
    #         Argument `P` must be _softmaxed_, P_logit is the logits of P (i.e. before softmaxing)! 
    #         Argument `ds` must be a dataset with position tracking. """
    #     losses, weights = self.get_profile_losses(ds, P, P_logit)
    #     return tf.reduce_mean( tf.multiply(losses, weights), axis=1 ) # (U)
    


    # def get_profile_scores(self, ds, P):
    #     """ Return for each profile the max score reached per batch, shape (U, x) where x is 
    #         number_of_batches * batch_size.
    #         Argument `P` must be _softmaxed_, don't pass the logits! """
    #     U = P.shape[-1]
    #     scores = None
    #     for batch, _ in ds:
    #         for X in batch:
    #             assert len(X.shape) == 5, str(X.shape)
    #             S, _, _ = self.call(X, P)        # shape (ntiles, N, U)
    #             S = tf.reduce_max(S, axis=(0,1)) # shape (U)
    #             if scores is None:
    #                 scores = tf.expand_dims(S, -1)
    #             else:
    #                 scores = tf.concat([scores, tf.expand_dims(S, -1)], axis=1)
                                    
    #     return scores
    


    # def profile_cleanup(self, pIdx: int, epoch: int):
    #     """ Add profile at pIdx to report profiles, mask match sites, and get newly initialized profiles """
    #     # get k+2s-mer, extract all k-mers, temporarily set k-mers as new profiles
    #     P = self.P_logit # shape: (k+2s, alphabet_size, U)
    #     b = P[:,:,pIdx].numpy() # type: ignore
    #     Pk_logit = np.empty(shape=(self.setup.k, self.data.alphabet_size(), (2*self.setup.s)+1), dtype=np.float32)
    #     for s in range(b.shape[0]-self.setup.k+1):
    #         Pk_logit[:,:,s] = b[s:(s+self.setup.k),:]
            
    #     Pk = tf.nn.softmax(Pk_logit, axis=1, name="Pk") # shape: (k, alphabet_size, 2s+1 -> U')
        
    #     # get best k-mer and report (unless it is the first or last k-mer when shift > 0)
    #     scores = tf.reduce_max( self.get_profile_scores(self.data.getDataset(), P = Pk), axis=1 )   # (U', x) -> (U')
    #     bestIdx = tf.math.argmax(scores, axis=0).numpy()
    #     threshold = self.setup.match_score_factor * scores.numpy()[bestIdx]
    #     losses, _ = self.get_profile_losses(self.data.getDataset(withPosTracking=True), Pk, Pk_logit)
    #     minloss = tf.reduce_min(losses[bestIdx,:]).numpy() # type: ignore
    #     sites, sitescores = self.get_profile_match_sites(self.data.getDataset(withPosTracking = True), 
    #                                                      Pk, threshold, bestIdx)
    #     if bestIdx not in [0, Pk.shape[2]-1] or self.setup.s == 0: # type: ignore
    #         returnEdgeCase = False
    #         # report the best k-profile
    #         self.profile_report.addProfile(epoch, Pk[:,:,bestIdx].numpy(), pIdx, threshold, minloss,  # type: ignore
    #                                        sites.numpy(), sitescores.numpy()) # type: ignore
        
    #         # "remove" match sites from genomes, site: <genomeIdx, contigIdx, frameIdx, tileStartPos, T-k+1_idx, U_idx>
    #         for site in sites: # type: ignore
    #             if all(site.numpy()[:4] == [-1, -1, -1, -1]):
    #                 logging.warning(f"[model.profile_cleanup] >>> Attempted to mask {site=} in exhausted tile, " \
    #                                 +"skipping.")
    #                 continue

    #             matchseq = self.data.softmask(genome_idx=site[0].numpy(), 
    #                                           sequence_idx=site[1].numpy(),
    #                                           frame_idx=site[2].numpy(), 
    #                                           start_pos=site[3].numpy()+site[4].numpy(), 
    #                                           masklen=self.setup.k)
    #             if len(matchseq) != self.setup.k:
    #                 logging.warning(f"[model.profile_cleanup] >>> Match sequence has wrong length: {len(matchseq)}" \
    #                                 + f", expected {self.setup.k}. Site {site} seems out of bounds")
                    
    #         # for debugging purpose, report the whole k+2s-profile as well
    #         whole_scores = tf.reduce_max( self.get_profile_scores(self.data.getDataset(), self.getP()), axis=1 ).numpy()
    #         whole_losses, _ = self.get_profile_losses(self.data.getDataset(withPosTracking=True), 
    #                                                   self.getP(), self.P_logit)
    #         self.whole_profile_report.addProfile(epoch, self.getP().numpy()[:,:,pIdx], pIdx,  # type: ignore
    #                                              self.setup.match_score_factor * whole_scores[pIdx],
    #                                              tf.reduce_min(whole_losses[pIdx,:]).numpy()) # type: ignore
            
    #     else:
    #         returnEdgeCase = True
    #         logging.info("[model.profile_cleanup] >>> Profile is an edge case, starting over")
    #         self.discarded_profile_report.addProfile(epoch, Pk[:,:,bestIdx].numpy(), pIdx, threshold, minloss,  # type: ignore
    #                                                  sites.numpy(), sitescores.numpy()) # type: ignore
    #         if self.P_logit_init is not None:
    #             # otherwise get stuck with this profile
    #             self.P_logit_init[:,:,pIdx] = np.ones((self.P_logit_init.shape[0], self.P_logit_init.shape[1]), 
    #                                                   dtype=np.float32) * np.min(self.P_logit_init)
            
    #     # reset profiles
    #     if self.P_logit_init is None:
    #         self.P_logit.assign(self._getRandomProfiles())
    #     else:
    #         self.P_logit.assign(self.P_logit_init)
            
    #     return returnEdgeCase
    


    # def get_profile_match_sites(self, ds, P, score_threshold, pIdx: int = None): # type: ignore
    #     """
    #     Get sites in the dataset where either all or a specific profile match according to a score threshold
    #         Parameters:
    #             ds: tf dataset
    #             P: profile tensor, shape (k[+2s], alphabet_size, U), needs to be softmaxed! Don't pass the logits!
    #             score_threshold (float or tensor): matching sites need to achieve at least this score
    #             pIdx (int): optional index of a single profile, if given only matching sites of that profile are 
    #                           reported
                
    #         Returns:
    #             sites (tensor): tensor of shape (X, 6) where X is the number of found sites and the second dimension
    #                             contains tuples with (genomeIdx, contigIdx, frameIdx, tileStartPos, tilePos, profileIdx)
    #             scores (tensor): tensor of shape (X, 1) containing the scores of the found sites
    #     """        
    #     score_threshold = tf.convert_to_tensor(score_threshold, dtype=tf.float32)
    #     assert score_threshold.shape in [(), (P.shape[-1])], f"{score_threshold=}, {score_threshold.shape=}"
            
    #     sites = None
    #     scores = None
    #     for batch in ds:
    #         X_b = batch[0]        # (B, tilePerX, N, f, tileSize, alphabetSize)
    #         posTrack_b = batch[1] # (B, tilePerX, N, f, <genomeIdx, contigIdx, frameIdx, TileStartPos>)
    #         assert len(X_b.shape) == 6, str(X_b.shape)
    #         assert posTrack_b.shape != (1, 0), f"{posTrack_b.shape=} -- use batch dataset with position tracking!"
    #         assert X_b.shape[0:4] == posTrack_b.shape[0:4], f"{X_b.shape} != {posTrack_b.shape}"
    #         for b in range(X_b.shape[0]): # iterate samples in batch
    #             # get profile match scores, i.e. the sum of the element-wise multiplication of each profile 
    #             #   at each sequence position in X --> Z
    #             X = X_b[b]                # (tilePerX, N, f, tileSize, alphabetSize)
    #             posTrack = posTrack_b[b]  # (tilePerX, N, f, <genomeIdx, contigIdx, frameIdx, TileStartPos>)
    #             _, _, Z = self.call(X, P) # (tilePerX, N, f, T-k+1, U)
    #             if pIdx is not None:
    #                 Z = Z[:,:,:,:,pIdx:(pIdx+1)] # only single profile, but keep dimensions

    #             # identify matches, i.e. match score >= score_threshold
    #             M = tf.greater_equal(Z, score_threshold) # (tilesPerX, N, f, T-k+1, U)

    #             # TODO: to use this in cleanup during training, add an argument to the function to switch this on/off
    #             #       also, if this prevents an OOM during profile cleanup, we also need to use this in the remaining
    #             #       code for evaluation etc, otherwise we might get an OOM there and runs still crash.
    #             # # get a tensor of same shape as Z where only the argmax of dimension 3 (T-k+1) is True
    #             # # not ideal as O(memory) might still be worst case if all values are the same, 
    #             # #   but let's hope this never happens
    #             # M = tf.logical_and(M, tf.equal(Z, tf.reduce_max(Z, axis=3, keepdims=True))) # (tilesPerX, N, f, T-k+1, U)

    #             # index tensor -> 2D tensor with shape (sites, 5) where each row is a match and the columns are indices:
    #             I = tf.cast(tf.where(M), tf.int32)       # (sites, <tilesPerX_idx, N_idx, f_idx, T-k+1_idx, U_idx>)

    #             # build the sites and scores tensors (tensorflow.org/versions/r2.10/api_docs/python/tf/gather_nd)
    #             _scores = tf.gather_nd(Z, I)                  # (sites, <score>)
    #             _sites = tf.gather_nd(posTrack, I[:,:3])      # (sites, <g,c,f,tspos>) # type: ignore
    #             _sites = tf.concat([_sites, I[:,3:]], axis=1) # (sites, <g,c,f,tspos,T-k+1_idx,U_idx>) # type: ignore

    #             if sites is None:
    #                 sites = _sites
    #                 scores = _scores
    #             else:
    #                 sites = tf.concat([sites, _sites], axis=0)
    #                 scores = tf.concat([scores, _scores], axis=0)

    #     if sites is None:
    #         return tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.float32)
        
    #     return sites, scores