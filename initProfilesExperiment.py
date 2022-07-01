import dataset
import sequtils as su

import numpy as np

def addAA(combinations: list):
    aa_alphabet = su.aa_alphabet[1:]
    newCombinations = []
    for aa in aa_alphabet:
        for c in combinations:
            newCombinations.append(c+aa)
            
    return newCombinations



def getExhaustiveProfiles(k, Q, mid_factor=1, bg_factor = 0, exp=3):
    aa_alphabet = su.aa_alphabet[1:]
    assert len(Q) == len(aa_alphabet), "[ERROR] >>> Q must have "+str(len(aa_alphabet))+" elements"
    assert mid_factor > 0, "[ERROR] >>> mid_factor must be > 0"
    assert bg_factor >= 0, "[ERROR] >>> bg_factor must be >= 0"
    assert exp >= 1, "[ERROR] >>> exp must be >= 1"
    #assert sigma > 0, "[ERROR] >>> sigma must be > 0"
    assert k >= exp, "[ERROR] >>> k must be >= exp"
    
    lflank = (k-exp)//2
    U = len(aa_alphabet)**exp
    
    # as middle three positions, use all 9.261 kombinations of AA, flanked by simply Q
    Q = np.array(Q, dtype=np.float32)
    profiles = np.repeat([Q], repeats=k, axis=0)
    bgmid = np.repeat([Q], repeats=exp, axis=0) * bg_factor
    #print("[DEBUG] >>> profiles.shape:", profiles.shape)
    profiles = np.repeat([profiles], repeats=U, axis=0)
    #print("[DEBUG] >>> profiles.shape:", profiles.shape)
    profiles = np.transpose(profiles, (1,2,0))
    #print("[DEBUG] >>> profiles.shape:", profiles.shape)
    #print("[DEBUG] >>> profiles[:,:,0]:", profiles[:,:,0])
    
    combinations = aa_alphabet
    if exp > 1:
        for _ in range(exp-1):
            combinations = addAA(combinations)
    
    assert len(combinations) == U, "[ERROR] >>> U ("+str(U)+") != len(combinations) ("+str(len(combinations))+")"
    for u in range(U):
        mid = dataset.oneHot(combinations[u]) * mid_factor
        profiles[lflank:lflank+exp,:,u] = (mid + bgmid)
        
    #u = 0
    #for a1 in su.aa_alphabet[1:]:
    #    for a2 in su.aa_alphabet[1:]:
    #        for a3 in su.aa_alphabet[1:]:
    #            mid = dataset.oneHot(''.join([a1,a2,a3]))
    #            #print("[DEBUG] >>> mid.shape:", mid.shape)
    #            #print("[DEBUG] >>> mid:      ", mid)
    #            profiles[lflank:lflank+3,:,u] = mid
    #            #print("[DEBUG] >>> profiles[:,:,u]:", profiles[:,:,u])
    #            u += 1
    #            #assert False
                
    return profiles



def getCustomMidProfiles(midSeqs: list, k, Q, mid_factor=1, bg_factor = 0):
    aa_alphabet = su.aa_alphabet[1:]
    assert max([len(m) for m in midSeqs]) <= k, "[ERROR] >>> mid element lengths cannot exceed k"
    assert len(Q) == len(aa_alphabet), "[ERROR] >>> Q must have "+str(len(aa_alphabet))+" elements"
    assert mid_factor > 0, "[ERROR] >>> mid_factor must be > 0"
    assert bg_factor >= 0, "[ERROR] >>> bg_factor must be >= 0"
    
    U = len(midSeqs)
    
    # as middle three positions, use all 9.261 kombinations of AA, flanked by simply Q
    Q = np.array(Q, dtype=np.float32)
    profiles = np.repeat([Q], repeats=k, axis=0)
    #print("[DEBUG] >>> profiles.shape:", profiles.shape)
    profiles = np.repeat([profiles], repeats=U, axis=0)
    #print("[DEBUG] >>> profiles.shape:", profiles.shape)
    profiles = np.transpose(profiles, (1,2,0))
    #print("[DEBUG] >>> profiles.shape:", profiles.shape)
    #print("[DEBUG] >>> profiles[:,:,0]:", profiles[:,:,0])
    
    for u in range(U):
        exp = len(midSeqs[u])
        lflank = (k-exp)//2
        bgmid = np.repeat([Q], repeats=exp, axis=0) * bg_factor
        mid = dataset.oneHot(midSeqs[u]) * mid_factor
        profiles[lflank:lflank+exp,:,u] = (mid + bgmid)
                
    return profiles