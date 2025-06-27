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


def get_background_model(order: int, model_type: str = "uniform", src: Path | list[str] = None, seed: int = None):
    """
    Get a background model for a given order and model type. The model can be either
    'uniform', 'random', 'data' or 'augustus'. The uniform model is a uniform distribution of nucleotides,
    while the data model is based on the dinucleotide frequencies from a given file.
    The Augustus model is based on the dinucleotide frequencies from Augustus for
    human intergenic regions and is restricted to order 1.
    The function returns a tuple of the model (np.ndarray) and the frequencies of the nucleotides (dict)
    and the alphabet used (str 'ACGT').
    """
    assert order >= 0, "Order must be greater than or equal to 0"
    assert model_type in ["uniform", "random", "data", "augustus"], \
        f"Model type must be either 'uniform', 'random', 'data' or 'augustus'. Got {model_type} instead."

    if model_type == "uniform":
        if src is not None:
            print(f"[Warning] >>> Source file {src} is ignored for uniform model")
        return _get_uniform_model(order)
    
    elif model_type == "augustus":
        assert order == 1, "Order must be 1 for Augustus model"
        if src is not None:
            print(f"[Warning] >>> Source file {src} is ignored for uniform model")
        return _get_augustus_model()
    
    elif model_type == "random":
        if src is not None:
            print(f"[Warning] >>> Source file {src} is ignored for random model")
        if seed is not None:
            np.random.seed(seed)
            # logging.debug(f"[DEBUG] >>> Setting random seed to {seed}")
        alphabet = "ACGT"
        size = len(alphabet)
        k = order + 1
        model = np.zeros((size,)*k)
        freqs = {}
        for i in range(size**order):
            # get indices of first k-1 dimensions of model
            j_idcs = np.unravel_index(i, (size,)*order)
            # logging.debug(f"[DEBUG] >>> {i=}, {j_idcs=}, {size=}, {k=}, {model.shape=}")
            abs_freq = np.random.uniform(0, 1, size=4)
            rel_freq = abs_freq / abs_freq.sum() # normalize frequencies to sum to 1
            model[j_idcs] = rel_freq # fill the last dimension with random frequencies
        
        # get k-mer frequencies, ATTENTION: seems not to yield the correct results for order > 0, best ignore the result
        if order > 0:
            eq_dist = np.linalg.matrix_power(model, 50)
            pair_prob = eq_dist * model

            freqs_arr = pair_prob.reshape((size**k))
            freqs_arr = freqs_arr / freqs_arr.sum() # normalize frequencies to sum to 1
        else:
            freqs_arr = model.reshape((size,)) # for 0-mers, i.e. the frequencies of the nucleotides

        freqs = {}
        for i in range(size**k):
            kmer = "".join([alphabet[c] for c in np.unravel_index(i, (size,)*k)]) # construct the k-mer from the indices
            freqs[kmer] = freqs_arr[i]
        return model, freqs, alphabet
    
    elif model_type == "data":
        alphabet = "ACGT"
        assert src is not None, "Source file or list of sequence strings must be provided for data model"
        assert isinstance(src, (Path, list)), "Source must be a Path or a list of sequence strings"
        if isinstance(src, Path):
            assert src.exists(), f"Source file {src} does not exist"
            records = [str(r.seq) for r in SeqIO.parse(src, "fasta")]
            assert len(records) > 0, f"Source file {src} is empty or contains no records"
            assert all(len(r) > 0 for r in records), f"Source file {src} contains empty sequences"
        else:
            records = src
            assert len(records) > 0, f"Source list is empty"
            assert all(isinstance(r, str) for r in records), \
                f"Source list must contain strings, got {set([type(r) for r in records])} instead"
            assert all(len(r) > 0 for r in records), f"Source list contains empty sequences"
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
        for seq in records:
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
                for j in range(size):
                    kmer = o_nt + alphabet[j]
                    idcs = o_idcs + (j,)
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
import tensorflow_text as tftext

class TrainedQ(tf.keras.Model): # type: ignore
    def __init__(self, 
                 data: ModelDataSet.ModelDataSet,
                 num_models: int = 2, # K
                 order: int = 0, # k-1, i.e. order of the models
                 rand_seed: int = None, **kwargs): # type: ignore
        """
        Set up model and most metaparamters
            Parameters:
                data (ModelDataSet.ModelDataSet): dataset to use for training and scanning
                num_models (int): number of models to train, i.e. K
                order (int): order of the models, i.e. k-1, where k is the k-mer length
                rand_seed (int): optional set a seed for tensorflow's rng
        """
        super().__init__(**kwargs)

        assert num_models > 0, f"Number of models must be greater than 0, got {num_models}"
        assert order >= 0, f"Order must be greater than or equal to 0, got {order}"

        # setting random seeds if desired
        self.nprng = np.random.default_rng(rand_seed) # if rand_seed is None, unpredictable entropy is pulled from OS
        if rand_seed is not None:
            logging.debug(f"[model.__init__] >>> setting tf global seed to {rand_seed}")
            tf.random.set_seed(rand_seed)

        self.data = data
        self.num_models = num_models
        self.order = order # order of the models, i.e. k-mer-length - 1 # TODO: adapt usage of this in model.py!
        self.k_dims = self.order + 1 # number of dimensions for k-mer encoding
        self.alphabet = data.alphabet # alphabet used for the model, e.g. ['A', 'C', 'G', 'T'] for DNA

        self.Q_logit = tf.Variable(tf.random.uniform(shape=(self.num_models,)+(len(self.alphabet),)*(self.order+1),
                                                     minval=-0.1, maxval=0.1, dtype=tf.float32), 
                                   trainable=True, name="Q_logit") # shape: (K,)+(alphabet_size,)*(order+1)
        self.m_logit = tf.Variable(tf.random.uniform(shape=(self.num_models,),
                                                     minval=-0.1, maxval=0.1, dtype=tf.float32), 
                                   trainable=True, name="m_logit") # shape: (K,)
        logging.debug(f"[background_model.__init__] >>> Q_logit shape: {self.Q_logit.shape}, m_logit shape: {self.m_logit.shape}")

        self.opt = tf.keras.optimizers.Adam(learning_rate=float(2))


    def getQ(self):
        """ Returns softmaxed Q (K,)+(alphabet_size,)*(order+1). """
        Q = tf.nn.softmax(self.Q_logit, axis=-1, name="Q")
        return Q


    def getM(self):
        """ Returns softmaxed m (K,). """
        m = tf.nn.softmax(self.m_logit, axis=0, name="m")
        return m


    def call(self, X):
        """ Input X is a tensor of shape (batch_size, ntiles, N, f, tile_size)+(alphabet_size,)*(order+1) """
        Q = self.getQ() # shape: (K,)+(alphabet_size,)*(order+1)
        m = self.getM() # shape: (K,)

        batch_dims = len(X.shape) - (1+self.k_dims) # number of batch dimensions, e.g. 4 for (batch_size, ntiles, N, f, tile_size)+(alphabet_size,)*(order+1)

        Qr = tf.reshape(Q, ((1,)*batch_dims)+(1,)+Q.shape) # shape: (1,          1,      1, 1,         1, K)+(alphabet_size,)*(order+1)
        X1 = tf.expand_dims(X, -(self.k_dims+1)) #           shape: (batch_size, ntiles, N, f, tile_size, 1)+(alphabet_size,)*(order+1)
        C = tf.multiply(X1, Qr) # shape: (batch_size, ntiles, N, f, tile_size, K)+(alphabet_size,)*(order+1)
        C1 = tf.reduce_sum(C, axis=list(range(-self.k_dims,0))) # shape: (batch_size, ntiles, N, f, tile_size, K), sum over alphabet_size dimensions
        D = tf.math.log(tf.maximum(C1, 1e-10)) # shape: (batch_size, ntiles, N, f, tile_size, K), log to avoid numerical issues
        D1 = tf.reduce_sum(D, axis=-2) # shape: (batch_size, ntiles, N, f, K), sum over tile_size dimension
        M = tf.reduce_max(D1, axis=-1) # shape: (batch_size, ntiles, N, f), max over K dimension
        D2 = D1 - tf.expand_dims(M, -1) # shape: (batch_size, ntiles, N, f, K), subtract max to avoid numerical issues
        D3 = tf.multiply(tf.exp(D2), m) # shape: (batch_size, ntiles, N, f, K), multiply by model weights
        D4 = tf.reduce_sum(D3, axis=-1) # shape: (batch_size, ntiles, N, f), sum over K dimension
        D5 = tf.math.log(tf.maximum(D4, 1e-10)) # shape: (batch_size, ntiles, N, f), log to avoid numerical issues
        S = tf.add(M, D5) # shape: (batch_size, ntiles, N, f), add max back to get the final score
        # logging.debug(f"[background_model.call] >>> Q:\n{Qt}\n\nX:\n{X}\n\nC:\n{C}\n\nC1:\n{C1}\n\nD:\n{D}\n\nD1:\n{D1}\n\nM:\n{M}\n\nD2\n{D2}\n\nm:\n{m}\n\nD3\n{D3}\n\nD4\n{D4}\n\nD5\n{D5}\n\nS:\n{S}")
        # logging.debug(f"[background_model.call] >>> P:\n{tf.reduce_sum(S)}\n\nloss:\n{-tf.math.log(tf.reduce_sum(S))}") # preview on loss

        return S


    def lossfun(self, S):
        P = tf.reduce_sum(S)
        return -P # P already in log space


    @tf.function()
    def train_step(self, X):
        with tf.GradientTape() as tape:
            S = self.call(X)
            loss = self.lossfun(S)
            
        grad = tape.gradient(loss, [self.Q_logit, self.m_logit])
        # logging.debug(f"[background_model.train_step] >>> S:\n{S}\n\nloss:\n{loss}\n\ngrad:\n{grad}")
        # logging.debug(f"[background_model.train_step] >>> grad shape: {[g.shape for g in grad]}")
        self.opt.apply(grad, [self.Q_logit, self.m_logit])
        
        return S, loss#, grad


    def train(self, lr=2, epochs=50, verbose=True, verbose_freq=10):
        def setLR(learning_rate):
            logging.debug(f"[model.train.setLR] >>> Setting learning rate to {learning_rate}")
            self.opt.learning_rate.assign(learning_rate)

        max_epochs = epochs
        steps_per_epoch = self.data.getStepsPerEpoch() # use the steps_per_epoch from the dataset, this should be accurate
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
            ds_train = self.data.getDataset(k = self.order+1, flatten_kmers=False, repeat = True)
            _bshape = None
            for batch, _ in ds_train: # shape: (batchsize, ntiles, N, f, tile_size, alphabet_size**(order+1)
                assert len(batch.shape) == 5+self.k_dims, f"Expected shape of length {5+self.k_dims}, got {batch.shape}"
                _bshape = batch.shape
                self.train_step(batch)

                steps += 1
                if steps >= steps_per_epoch:
                    break
                    
            lossls = []
            ds_loss = self.data.getDataset(k = self.order+1, flatten_kmers=False, repeat = False)
            for batch, _ in ds_loss: # shape: (batchsize, ntiles, N, f, tile_size, alphabet_size**(order+1)
                assert len(batch.shape) == 5+self.k_dims, f"Expected shape of length {5+self.k_dims}, got {batch.shape}"
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



    def scan_data(self, window_size: int):
        """ Perform a scan over the data and return the scores. This is needed for training the SpecificProfile model.
        Args:
            window_size (int): size of the sliding window to use for scanning the data
        Returns:
            np.ndarray: scores for each tile in the dataset, shape (batches, ntiles, N, f, tile_size-window_size+1, K)
        """ 
        assert window_size > 0, f"Window size must be greater than 0, got {window_size}"
        assert window_size <= self.data.tile_size, \
            f"Window size {window_size} must be less than or equal to tile size {self.data.tile_size}"
        
        ds = self.data.getDataset(k = self.order+1, flatten_kmers=False, repeat = False)
        Qsm = tf.maximum(self.getQ(), 1e-10)  # shape: (K,)+(alphabet_size,)*(order+1), avoid numerical issues in log
        Q = tf.math.log(Qsm)
        
        ts_dim = -(self.k_dims + 1) # tile_size dimension index in X
        scores = []
        for batch, _ in ds:
            assert len(batch.shape) == 5+self.k_dims, f"Expected shape of length {5+self.k_dims}, got {batch.shape}"
            # note: the tile_size for order >= 1 is already reduced by k-1
            # Xk shape (batch_size, ntiles, N, f, tile_size-k+1)+(alphabet_size,)*(order+1)
            # add `order` positions of zeros to the beginning of the sequence
            shape = list(batch.shape)
            shape[ts_dim] = self.order
            Xz = tf.zeros(tuple(shape), dtype=batch.dtype) # shape: (batch_size, ntiles, N, f, order)+(alphabet_size,)*(order+1)
            X = tf.concat([Xz, batch], axis=ts_dim)
            assert X.shape[ts_dim] == self.data.tile_size, \
                f"Expected tile size {self.data.tile_size} in X, got {X.shape=}"
            
            batch_dims = len(X.shape) - (1+self.k_dims) # number of batch dimensions, e.g. 4 for (batch_size, ntiles, N, f, tile_size)+(alphabet_size,)*(order+1)
            Q1 = tf.reshape(Q, ((1,)*batch_dims)+(1,)+Q.shape) # shape: (         1,      1, 1, 1,         1, K)+(alphabet_size,)*(order+1)
            X1 = tf.expand_dims(X, -(self.k_dims+1)) #           shape: (batch_size, ntiles, N, f, tile_size, 1)+(alphabet_size,)*(order+1)
            R = tf.multiply(X1, Q1) # shape: (batch_size, ntiles, N, f, tile_size, K)+(alphabet_size,)*(order+1)
            assert R.shape == X.shape[:ts_dim] + (self.data.tile_size,) + Q.shape, \
                f"Expected shape {X.shape[:ts_dim] + (self.data.tile_size,) + Q.shape}, got {R.shape}"
            R1 = tf.reduce_sum(R, axis=list(range(-self.k_dims,0))) # shape: (batch_size, ntiles, N, f, tile_size, K), sum over alphabet_size dimensions
            
            # at this point, each kmer in X was multiplied with Q and the result is in R1
            # now we need to slide the window over the tile_size dimension
            # luckily, tensorflow has this built-in
            S = tftext.sliding_window(data=R1, width=window_size, axis=-2) # shape: (batch_size, ntiles, N, f, tile_size-width+1, width, K)
            # logging.debug(f"[background_model.scan_data] >>> S shape after sliding window: {S.shape}")
            S = tf.reduce_sum(S, axis=-2) # shape: (batch_size, ntiles, N, f, tile_size-width+1, K), sum over new width dimension
            # logging.debug(f"[background_model.scan_data] >>> S shape after sum: {S.shape}")
            assert S.shape[-2:] == (self.data.tile_size - window_size + 1, self.num_models), \
                f"Expected ({self.data.tile_size - window_size + 1}, {self.num_models}) in scores, got {S.shape}"

            scores.append(S) # append the scores for this batch

        scores = tf.concat(scores, axis=0) # concatenate the scores for all batches
        # logging.debug(f"[background_model.scan_data] >>> Final scores shape: {scores.shape}")
        return scores.numpy() # convert to numpy array and return