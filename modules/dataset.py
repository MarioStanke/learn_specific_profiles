import numpy as np
from numpy.lib.utils import source
import tensorflow as tf
from tqdm import tqdm
import sequtils as su

# translate single DNA sequence to AA sequence
def sequence_translation(S, rc = False):
    if rc:
        S = S[::-1].translate(su.rctbl)
        
    prot = ""
    for i in range(0, len(S)-3+1, 3):
        codon = S[i:i+3]
        if codon not in su.genetic_code: # real sequences may contain N's or softmasking or ambiguous bases
            prot += ' '                  # use null aa in that case
        else:
            prot += su.genetic_code[codon]
            
    return prot
    
    
    
# allow for custom frame offsets, see 20211011_profileFindingBatchTranslationSketch2.png 
def three_frame_translation(S, rc = False, offsets=range(3)):
    T = []
    if rc:
        S = S[::-1].translate(su.rctbl)

    for f in offsets: # frame
        prot = ""
        for i in range(f, len(S)-3+1, 3):
            codon = S[i:i+3]
            if codon not in su.genetic_code: # real sequences may contain N's or softmasking or ambiguous bases
                prot += ' '                  # use null aa in that case
            else:
                prot += su.genetic_code[codon]
        T.append(prot)

    return T



def translateSequenceTiles(sequence, fwd_a, rc_b, tilesize):
    assert fwd_a >= 0, str(fwd_a)+" < 0"
    assert rc_b > 0, str(rc_b)+" <= 0"
    assert tilesize % 3 == 0, str(tilesize)+" % 3 not 0 ("+str(tilesize%3)+")"
    
    seqlen = len(sequence)
    assert fwd_a < seqlen, str(fwd_a)+", "+seqlen
    assert rc_b <= seqlen, str(rc_b)+", "+seqlen
    
    fwd_b = min(seqlen, fwd_a+tilesize+2) # add two extra positions for frames 1 and 2
    rc_a = max(0, rc_b-tilesize-2)
    
    fwd_seq = sequence[fwd_a:fwd_b]
    rc_seq = sequence[rc_a:rc_b]
    rc_seq = rc_seq[::-1].translate(su.rctbl)
    #print("DEBUG >>> Sequence:\n'"+sequence+"'")
    assert len(fwd_seq) == len(rc_seq), "\nfwd: "+str(fwd_a)+" : "+str(fwd_b)+" -> "+str(len(fwd_seq))+"\n rc: "+str(rc_a)+" : "+str(rc_b)+" -> "+str(len(rc_seq))+"\nseqlen: "+str(seqlen)+"\n'"+str(fwd_seq)+"'"
        
    aa_seqs = three_frame_translation(fwd_seq)
    aa_seqs.extend(three_frame_translation(rc_seq))
    if fwd_b == seqlen:
        assert rc_a == 0, str(rc_a)

    return aa_seqs, fwd_b == seqlen



def oneHot(aa_seq):
    I = np.eye(su.aa_alphabet_size + 1) # for numpy-style one-hot encoding
    x = su.to_idx(aa_seq, su.aa_idx)
    one_hot = I[x] # here still aa_alphabet_size + 1 entries
    one_hot = one_hot[:,1:] 
    return one_hot



# Use a generator to get genome batches, simplified position handling
def createBatch(ntiles, aa_tile_size: int, genomes, withPosTracking: bool = False):
    assert aa_tile_size >= 1, "aa_tile_size must be positive, non-zero (is: "+str(aa_tile_size)+")"
    tile_size = aa_tile_size * 3 # tile_size % 3 always 0
    N = len(genomes)
    state = [{'idx': 0, 'fwd_a': 0, 'rc_b': None, 'exhausted': (len(seqs) == 0)} for seqs in genomes]

    while not all(s['exhausted'] for s in state):
        X = np.zeros([ntiles, N, 6, aa_tile_size, su.aa_alphabet_size], dtype=np.float32)   # (tilesPerX, N, 6, T, 21)
        I = np.eye(su.aa_alphabet_size + 1) # for numpy-style one-hot encoding
        if withPosTracking:
            # [:,:,:,0] - genome IDs, [:,:,:,1] - contig IDs, [:,:,2] - start pos of X[:,:,:,0,:] w.r.t. genome sequence, [:,:,3] - aa seqlen in that tile and frame, -1: exhausted
            # fwd/rc start both refer to the 0-based position of the 0th position in the respective tile _on the fwd strand_
            posTrack = np.ones([ntiles, N, 6, 4], dtype=np.int32) *-1                          # (tilesPerX, N, 6, (genomeIDs, contigIDs, startPos, aa_seqlen))
        for t in range(ntiles):
            for g in range(N):
                if state[g]['exhausted']:
                    continue
                    
                sidx = state[g]['idx']
                slen = len(genomes[g][sidx])
                if state[g]['rc_b'] is None:
                    state[g]['rc_b'] = slen

                fwd_a = state[g]['fwd_a']
                fwd_b = min(slen, fwd_a+tile_size) # seq[fwd_a:fwd_b]   -> a>>>>>>>b
                rc_b = state[g]['rc_b']
                rc_a = max(0, rc_b-tile_size)      # rc(seq[rc_a:rc_b]) -> b<<<<<<<a
                if withPosTracking:
                    posTrack[t,g,:,0] = g
                    posTrack[t,g,:,1] = sidx
                    posTrack[t,g,0,2] = fwd_a
                    posTrack[t,g,1,2] = fwd_a+1
                    posTrack[t,g,2,2] = fwd_a+2
                    posTrack[t,g,3,2] = rc_b-1
                    posTrack[t,g,4,2] = rc_b-2
                    posTrack[t,g,5,2] = rc_b-3
                    
                assert sidx < len(genomes[g]), str(sidx)+" >= "+str(len(genomes[g]))+" for genome "+str(g)
                
                # translate and add tiles
                sequence = genomes[g][sidx]
                if type(sequence) is not str:
                    sequence = tf.compat.as_str(sequence) # with tf, input are byte-strings and need to be converted back
                
                aa_seqs, seqExhausted = translateSequenceTiles(sequence, fwd_a, rc_b, tile_size)
                for frame in range(6):
                    aa_seq = aa_seqs[frame]
                    assert len(aa_seq) <= aa_tile_size, str(len(aa_seq))+" != "+str(aa_tile_size)+", fwd_a, rc_b, slen, tile, genome, frame: "+str((fwd_a, rc_b, slen, t, g, frame))                        
                    x = su.to_idx(aa_seq, su.aa_idx)
                    num_aa = x.shape[0]
                    if withPosTracking:
                        posTrack[t,g,frame,3] = num_aa
                        
                    if (num_aa > 0):
                        one_hot = I[x] # here still aa_alphabet_size + 1 entries
                        # missing sequence will be represented by an all-zero vector
                        one_hot = one_hot[:,1:] 
                        X[t,g,frame,0:num_aa,:] = one_hot
                        
                # update state
                state[g]['fwd_a'] = fwd_b
                state[g]['rc_b'] = rc_a
                if seqExhausted:
                    state[g]['idx'] += 1
                    state[g]['fwd_a'] = 0
                    state[g]['rc_b'] = None
                    
                if state[g]['idx'] == len(genomes[g]):
                    state[g]['exhausted'] = True
        
        if withPosTracking:
            yield X, posTrack
        else:
            yield X, []



def getDataset(genomes,
               tiles_per_X: int,
               tile_size: int,
               withPosTracking: bool = False):
    if withPosTracking:
        ds = tf.data.Dataset.from_generator(
            createBatch,
            args = (tf.constant(tiles_per_X), tf.constant(tile_size), tf.constant(genomes, dtype=tf.string), 
                    tf.constant(True)),

            # vvv used in newer versions of TF vvv
            # output_signature = (tf.TensorSpec(shape = ([batch_size, len(genomes), 6, tile_size, su.aa_alphabet_size], 
            #                                            [batch_size, len(genomes), 2]),
            #                                   dtype = (tf.float32, tf.int32))

            # vvv deprecated in newer versions of TF vvv
            output_types = (tf.float32, tf.int32),
            output_shapes = (tf.TensorShape([tiles_per_X, len(genomes), 6, tile_size, su.aa_alphabet_size]),
                             tf.TensorShape([tiles_per_X, len(genomes), 6, 4]))
        )
    else:
        ds = tf.data.Dataset.from_generator(
            createBatch,
            args = (tf.constant(tiles_per_X), tf.constant(tile_size), tf.constant(genomes, dtype=tf.string), 
                    tf.constant(False)),

            # vvv used in newer versions of TF vvv
            # output_signature = (tf.TensorSpec(shape = [batch_size, len(genomes), 6, tile_size, su.aa_alphabet_size],
            #                                   dtype = tf.float32))

            # vvv deprecated in newer versions of TF vvv
            output_types = (tf.float32, tf.float32),
            output_shapes = (tf.TensorShape([tiles_per_X, len(genomes), 6, tile_size, su.aa_alphabet_size]),
                             tf.TensorShape(0))
        )
    return ds



class DatasetHelper:
    """ Helper class to easily pass dataset parameters around and tweak dataset creation as needed """
    
    def __init__(self, 
                 genomes,
                 tiles_per_X: int,
                 tile_size: int,
                 batch_size: int = None,
                 prefetch: int = None):
        """ If batch_size and/or prefetch are None (default), 
            no batch size and/or prefetch dataset is created unless
            specified in the call to DatasetHelper.getDataset() """
        self.genomes = genomes
        self.tilesPerX = tiles_per_X
        self.tileSize = tile_size
        self.batchSize = batch_size
        self.prefetch = prefetch
        
    def getDataset(self, 
                   tiles_per_X: int = None,
                   tile_size: int = None,
                   batch_size: int = None,
                   prefetch: int = None,
                   repeat: bool = False,
                   withPosTracking: bool = False):
        """ Any argument specified here overwrites the defaults from object constuction """
        tilesPerX = tiles_per_X if tiles_per_X is not None else self.tilesPerX
        tileSize = tile_size if tile_size is not None else self.tileSize
        batchSize = batch_size if batch_size is not None else self.batchSize
        prefetch_ = prefetch if prefetch is not None else self.prefetch
        ds = getDataset(self.genomes,
                        tilesPerX,
                        tileSize,
                        withPosTracking)
        
        if repeat:
            ds = ds.repeat()
        
        if batchSize is not None:
            ds = ds.batch(batchSize)
            
        if prefetch_ is not None:
            ds = ds.prefetch(prefetch_)
            
        return ds
    

    
#def restoreGenomePosition(aaTilePos, frame, fwdStart, rcStart, k):
#    assert frame in range(6), str(frame)+" must be in [0,5]"
#    p = aaTilePos*3 # to dna coord
#    if frame < 3:
#        p += frame     # add frame shift
#        p += fwdStart
#    else:
#        p += frame-3   # add frame shift
#        p = rcStart - p - (k*3) + 1
#
#    return p

def restoreGenomePosition(aaTilePos, start, k, fwd: bool):
    p = aaTilePos*3 # to dna coord
    if fwd:
        p += start
    else:
        p = start - p - (k*3) + 1

    return p


        
# **Test Batch Generator**
# Retranslate and assemble all batches, compare to translations of genome sequences

rev_aa_idx = dict((i,c) for i,c in enumerate(su.aa_alphabet))
def to_aa(onehot, dropEmptyColumns = True):
    assert onehot.shape[1] == 21, str(onehot.shape)
    aa_seq = ""
    for c in range(onehot.shape[0]):
        if dropEmptyColumns and np.max(onehot[c,:]) != 1:
            continue
        elif np.max(onehot[c,:]) != 1:
            aa_seq += ' '
        else:
            aa_idx = np.argmax(onehot[c,:])
            assert onehot[c, aa_idx] == 1, str(onehot[c,:])+", "+str(aa_idx)+", "+str(onehot[c,aa_idx])
            aa_idx += 1 # argmax + 1 as in translation, empty aa is cut out
            aa_seq += rev_aa_idx[aa_idx]
        
    return aa_seq



def testGenerator(genomes, ntiles, tile_size, limit = 10000):
    # limit test to subset of sequences for large genomes    
    if max([sum([len(s) for s in genome]) for genome in genomes]) > limit:
        testgenome = [[] for _ in range(len(genomes))]
        i = 0
        while max([sum([len(s) for s in genome]) for genome in testgenome]) < limit:
            for g in range(len(genomes)):
                if i < len(genomes[g]):
                    if len(genomes[g][i]) > limit:
                        testgenome[g].append(genomes[g][i][0:limit])
                    else:
                        testgenome[g].append(genomes[g][i])

            i += 1
    else:
        testgenome = [[s for s in genome] for genome in genomes]
        
    print("[INFO] >>> Test genome lengths:", [sum([len(s) for s in genome]) for genome in genomes])

    # translate and concatenate whole testgenome aa sequences
    genome_aa = [[""]*6 for _ in range(len(testgenome))]
    for g in range(len(testgenome)):
        for i in range(len(testgenome[g])):
            #aa_seqs = su.six_frame_translation(testgenome[g][i])
            aa_seqs = three_frame_translation(testgenome[g][i])
            aa_seqs.extend(three_frame_translation(testgenome[g][i], True))
            for f in range(len(aa_seqs)):
                genome_aa[g][f] += aa_seqs[f].replace(' ', '')

    X_to_genome_aa = [[""]*6 for _ in range(len(testgenome))] # for each genome and each frame, concatenate translated aa seqs

    # create generator
    Xgen = createBatch(ntiles, tile_size, testgenome)

    # iterate through generator, transforming and concatenating aa sequences
    # X.shape [ntiles, N, 6, tile_size, su.aa_alphabet_size]
    for X, _ in tqdm(Xgen):
        for t in range(X.shape[0]):
            for g in range(X.shape[1]):
                for f in range(X.shape[2]):
                    x_aa_seq = to_aa(X[t,g,f,:,:])
                    X_to_genome_aa[g][f] += x_aa_seq

    # compare aa sequences
    try:
        assert np.all(genome_aa == X_to_genome_aa), str(genome_aa)+"\n\n!=\n\n"+str(X_to_genome_aa)
        print("[INFO] >>> testGenerator - restore sequences: All good")
        
    except:
        # In case of failure, see where the sequences differ or if only frames were shifted (for some unknown reason)
        assert len(genome_aa) == len(X_to_genome_aa)
        for g in range(len(genome_aa)):
            assert len(genome_aa[g]) == len(X_to_genome_aa[g])
            for f in range(len(genome_aa[g])):
                if genome_aa[g][f] != X_to_genome_aa[g][f]:
                    if f < 3 or (not (genome_aa[g][f] == X_to_genome_aa[g][3] \
                                      or genome_aa[g][f] == X_to_genome_aa[g][4] \
                                      or genome_aa[g][f] == X_to_genome_aa[g][5])):
                        print("genome", g, "-- frame", f)
                        for i in range(len(X_to_genome_aa[g][f])):
                            if X_to_genome_aa[g][f][i] == genome_aa[g][f][i]:
                                print(X_to_genome_aa[g][f][i], genome_aa[g][f][i], i)

                            else:
                                print(X_to_genome_aa[g][f][i], genome_aa[g][f][i], i, "<--")

                        assert False
                        
        print("[INFO] >>> testGenerator - restore sequences: rc frames shifted, otherwise all good")

    # test position restoring
    XgenPos = createBatch(ntiles, tile_size, testgenome, True)
    k = 11
    for X, P in tqdm(XgenPos): # (tilesPerX, N, 6, T, 21), (tilesPerX, N, 6, (genomeIDs, contigIDs, startPos, aa_seqlen))
        for t in range(X.shape[0]):
            for g in range(X.shape[1]):
                for f in range(X.shape[2]):
                    for p in range(X.shape[3]-k+1):
                        if P[t,g,f,0] != -1:
                            assert P[t,g,f,0] == g, str(P[t,g,f,:])
                            c = P[t,g,f,1]
                            kmerOH = X[t,g,f,p:(p+k),:]
                            kmerAA = to_aa(kmerOH, False)
                            assert len(kmerAA) == k, str(k)+", '"+str(kmerAA)+"' ("+str(len(kmerAA))+")\n"+str(kmerOH)
                            pos = restoreGenomePosition(p, P[t,g,f,2], k, (f<3)) # restoreGenomePosition(aaTilePos, start, k, fwd: bool)
                            end = pos+(k*3)
                            if pos >= 0 and end <= len(testgenome[g][c]): # sometimes negative for rc frames or reaching over the sequence for fwd frames, skip
                                sourceKmer = testgenome[g][c][pos:end]
                                sourceKmerAA = sequence_translation(sourceKmer) if f < 3 else sequence_translation(sourceKmer, True)
                                #print("DEBUG >>> "+sourceKmerAA+" (source)\n          "+kmerAA+" (observed)\n"+str((g,c,t,f,pos)))
                                assert len(kmerAA) == len(sourceKmerAA), "\n'"+sourceKmerAA+"' !=\n'"+kmerAA+"'\n"+str((g,c,t,f,p,pos,P[t,g,f,:]))
                                assert kmerAA == sourceKmerAA, "\n'"+sourceKmerAA+"' !=\n'"+kmerAA+"'\n"+str((g,c,t,f,p,pos,P[t,g,f,:]))

    print("[INFO] >>> testGenerator - restore positions: All good")
