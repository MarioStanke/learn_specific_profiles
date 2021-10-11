import numpy as np
from tqdm import tqdm
import sequtils as su

# manually handle reverse complements, assume that len(S) = x*3 + 2
def three_frame_translation(S):
    assert len(S) % 3 == 2, str(len(S))+" % 3 = "+str(len(S)%3)+"\n\n'"+str(S)+"'"
    T = []
    for f in range(3): # frame
        prot = ""
        for i in range(f, len(S)-3+1, 3):
            codon = S[i:i+3]
            if codon not in su.genetic_code: # real sequences may contain N's or softmasking or ambiguous bases
                prot += ' '                  # use null aa in that case
            else:
                prot += su.genetic_code[codon]
        T.append(prot)
    return T



def translateSequences(fwdSeq, pre_rcSeq, startPadding: bool):
    rcSeq = pre_rcSeq[::-1].translate(su.rctbl)
    if startPadding:
        rcSeq += '  ' # add padding
        
    fwdSeq += ' '*(2-(len(fwdSeq)%3)) # add padding such that len%3==2
    rcSeq = ' '*(2-(len(rcSeq)%3)) + rcSeq

    # some nucleotides are part of both neighboring tiles
    aa_seqs_fwd = three_frame_translation(fwdSeq)
    aa_seqs_rc  = three_frame_translation(rcSeq)
    aa_seqs_fwd.extend(aa_seqs_rc)
    return aa_seqs_fwd



# Use a generator to get genome batches
def createBatch(ntiles, aa_tile_size: int, genomes):
    assert aa_tile_size >= 1, "aa_tile_size must be positive, non-zero (is: "+str(aa_tile_size)+")"
    tile_size = aa_tile_size * 3 # tile_size % 3 always 0
    N = len(genomes)
    state = [{'idx': 0, 'pos': 0, 'exhausted': (len(seqs) == 0)} for seqs in genomes]
    while not all(s['exhausted'] for s in state):
        X = np.zeros([ntiles, N, 6, aa_tile_size, su.aa_alphabet_size], dtype=np.float32)
        I = np.eye(su.aa_alphabet_size + 1) # for numpy-style one-hot encoding
        for t in range(ntiles):
            for i in range(N):
                if state[i]['exhausted']:
                    continue
                    
                sidx = state[i]['idx']
                slen = len(genomes[i][sidx])
                
                start = state[i]['pos']
                end = min(slen, start+tile_size)
                framestart = max(0, start-2)
                frameend = min(slen, end+2)
                assert start < end
                assert framestart < end
                
                # update state
                state[i]['pos'] = end
                if end == slen:
                    state[i]['idx'] += 1
                    state[i]['pos'] = 0
                    
                if state[i]['idx'] == len(genomes[i]):
                    state[i]['exhausted'] = True
                                
                # translate and add tiles
                sequence = genomes[i][sidx]
                if type(sequence) is not str:
                    sequence = str(sequence) # with tf, input are byte-strings and need to be converted back
                    
                aa_seqs = translateSequences(sequence[start:frameend], 
                                             sequence[framestart:end],
                                             (start == 0))
                for frame in range(6):
                    aa_seq = aa_seqs[frame]
                    assert len(aa_seq) <= aa_tile_size, str(len(aa_seq))+" != "+str(aa_tile_size)+", start, end, frameend, slen, tile, genome, frame: "+str((start, end, frameend, slen, t, i, frame))                        
                    x = su.to_idx(aa_seq, su.aa_idx)
                    num_aa = x.shape[0]
                    if (num_aa > 0):
                        one_hot = I[x] # here still aa_alphabet_size + 1 entries
                        # missing sequence will be represented by an all-zero vector
                        one_hot = one_hot[:,1:] 
                        X[t,i,frame,0:num_aa,:] = one_hot
                        
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
        
    return aa_seq#.lstrip().rstrip()



def testGenerator(genomes, tile_size, limit = 10000):
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

    # translate and concatenate whole testgenome aa sequences
    genome_aa = [[""]*6 for _ in range(len(testgenome))]
    for g in range(len(testgenome)):
        for i in range(len(testgenome[g])):
            aa_seqs = su.six_frame_translation(testgenome[g][i])
            for f in range(len(aa_seqs)):
                genome_aa[g][f] += aa_seqs[f].replace(' ', '')#.lstrip().rstrip()

    X_to_genome_aa = [[""]*6 for _ in range(len(testgenome))] # for each genome and each frame, concatenate translated aa seqs

    # create generator
    Xgen = createBatch(5, tile_size, testgenome)

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