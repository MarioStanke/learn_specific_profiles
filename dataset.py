import numpy as np
from tqdm import tqdm
import sequtils as su

# allow for custom frame offsets, see 20211011_profileFindingBatchTranslationSketch2.png 
def three_frame_translation(S, offsets=range(3)):
    #assert len(S) % 3 == 2, str(len(S))+" % 3 = "+str(len(S)%3)+"\n\n'"+str(S)+"'"
    T = []
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



def rcFrameOffsets(seqlen):
    sm3 = seqlen%3
    if sm3 == 0:
        return [2,0,1]
    elif sm3 == 1:
        return [1,2,0]
    else:
        return [0,1,2]
    
    
    
def translateSequences(sequence, start, tilesize):
    assert start >= 0, str(start)+" < 0"
    assert tilesize % 3 == 0, str(tilesize)+" % 3 not 0 ("+str(tilesize%3)+")"
    
    seqlen = len(sequence)
    assert start < seqlen, str(start)+", "+seqlen
    
    tileend = min(seqlen, start+tilesize+2)
    
    fwd_seq = sequence[start:tileend]
    rc_seq = fwd_seq[::-1].translate(su.rctbl)
    
    sm3 = seqlen%3
    tm3 = len(rc_seq)%3
    if sm3 == tm3: # last tile has same mod as sequence
        rc_offsets = range(3)
    else:
        rc_offsets = rcFrameOffsets(seqlen)
    
    
    aa_seqs = three_frame_translation(fwd_seq)
    aa_seqs.extend(three_frame_translation(rc_seq, rc_offsets))
    return aa_seqs



# Use a generator to get genome batches
def createBatch(ntiles, aa_tile_size: int, genomes, withPosTracking: bool = False):
    assert aa_tile_size >= 1, "aa_tile_size must be positive, non-zero (is: "+str(aa_tile_size)+")"
    tile_size = aa_tile_size * 3 # tile_size % 3 always 0
    N = len(genomes)
    state = [{'idx': 0, 'pos': 0, 'exhausted': (len(seqs) == 0)} for seqs in genomes]
    while not all(s['exhausted'] for s in state):
        X = np.zeros([ntiles, N, 6, aa_tile_size, su.aa_alphabet_size], dtype=np.float32)
        I = np.eye(su.aa_alphabet_size + 1) # for numpy-style one-hot encoding
        if withPosTracking:
            # [:,:,0] - contig IDs, [:,:,1] - tile start, [:,:,2] - tile size, -1: exhausted
            posTrack = np.ones([ntiles, N, 3], dtype=np.int32) *-1 
        for t in range(ntiles):
            for i in range(N):
                if state[i]['exhausted']:
                    continue
                    
                sidx = state[i]['idx']
                slen = len(genomes[i][sidx])
                sm3 = slen % 3
                start = state[i]['pos']
                end = min(slen, start+tile_size)
                if withPosTracking:
                    posTrack[t,i,0] = sidx
                    posTrack[t,i,1] = start
                    posTrack[t,i,2] = end-start
                
                # translate and add tiles
                sequence = genomes[i][sidx]
                if type(sequence) is not str:
                    sequence = str(sequence) # with tf, input are byte-strings and need to be converted back
                
                aa_seqs = translateSequences(sequence, start, tile_size)
                for frame in range(6):
                    aa_seq = aa_seqs[frame]
                    assert len(aa_seq) <= aa_tile_size, str(len(aa_seq))+" != "+str(aa_tile_size)+", start, end, slen, tile, genome, frame: "+str((start, end, slen, t, i, frame))                        
                    x = su.to_idx(aa_seq, su.aa_idx)
                    num_aa = x.shape[0]
                    if (num_aa > 0):
                        one_hot = I[x] # here still aa_alphabet_size + 1 entries
                        # missing sequence will be represented by an all-zero vector
                        one_hot = one_hot[:,1:] 
                        X[t,i,frame,0:num_aa,:] = one_hot
                        
                # update state
                state[i]['pos'] = end
                if end >= slen-sm3:
                    state[i]['idx'] += 1
                    state[i]['pos'] = 0
                    
                if state[i]['idx'] == len(genomes[i]):
                    state[i]['exhausted'] = True
        
        if withPosTracking:
            yield X, posTrack
        else:
            yield X


        
# **Test Batch Generator**
# Retranslate and assemble all batches, compare to translations of genome sequences

rev_aa_idx = dict((i,c) for i,c in enumerate(su.aa_alphabet))
def to_aa(onehot):
    assert onehot.shape[1] == 21, str(onehot.shape)
    aa_seq = ""
    for c in range(onehot.shape[0]):
        if np.max(onehot[c,:]) != 1:
            continue
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
            aa_seqs = su.six_frame_translation(testgenome[g][i])
            for f in range(len(aa_seqs)):
                if f < 3:
                    genome_aa[g][f] += aa_seqs[f].replace(' ', '')
                else:
                    genome_aa[g][f] = aa_seqs[f].replace(' ', '') + genome_aa[g][f]

    X_to_genome_aa = [[""]*6 for _ in range(len(testgenome))] # for each genome and each frame, concatenate translated aa seqs

    # create generator
    Xgen = createBatch(ntiles, tile_size, testgenome)

    # iterate through generator, transforming and concatenating aa sequences
    # X.shape [ntiles, N, 6, tile_size, su.aa_alphabet_size]
    for X in tqdm(Xgen):
        for t in range(X.shape[0]):
            for g in range(X.shape[1]):
                for f in range(X.shape[2]):
                    x_aa_seq = to_aa(X[t,g,f,:,:])
                    if f < 3:
                        X_to_genome_aa[g][f] += x_aa_seq
                    else:
                        X_to_genome_aa[g][f] = x_aa_seq + X_to_genome_aa[g][f]

    # compare aa sequences
    try:
        assert np.all(genome_aa == X_to_genome_aa), str(genome_aa)+"\n\n!=\n\n"+str(X_to_genome_aa)
        print("[INFO] >>> testGenerator: All good")
        
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
                        
        print("[INFO] >>> testGenerator: rc frames shifted, otherwise all good")