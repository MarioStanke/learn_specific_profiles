import json
import os

# define a list for each grid parameter (may contain one or more elements), later each possible combination is created
paramlists = {
    'tile_size': [334], # tile size measured in amino acids
    'batch_size': [1], # number of X to generate per batch
    'tiles_per_X': [13], # number of tiles per X (-> X.shape[0])
    'k': [20], # length of profiles
    's': [6], # shift to both sides
    'alpha': [1e-6], # loss norm, unused
    'gamma': [1], # softmax scale
    'l2': [0.01], # L2 reg factor
    'match_score_factor': [0.7],
    'learning_rate': [2],
    'rho': [0], # influence of initial sampling position on profile initialization
    'sigma': [1], # stddev of random normal values added to profile initialization (mean 0)
    'profile_plateau_dev': [150],
    'n_best_profiles': [2],
    'lossStrategy': ['experiment'],
    'mid_factor': [0.5], 
    'bg_factor': [0.5],
    'exp': [2]
}

grid = []
def createGrid():
    ...

# create all possible parameter combinations from seedFindingArgDict
def createParameterDicts(currentDictRef, parameterDictRef, parameterDictListRef):
    currentDict = dict(currentDictRef)
    parameterDict = dict(parameterDictRef)
    
    if parameterDict:
        key = list(parameterDict.keys())[0]
        values = np.array(parameterDict[key]).flatten().tolist() # for guaranteed iterate
        parameterDict.pop(key)  # remove key for recursion
        for value in values:
            if value or (isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0):   # may be None or False, numeric 0 is fine however
                currentDict[key] = value
            elif key in currentDict:
                currentDict.pop(key)
                    
            createParameterDicts(currentDict, parameterDict, parameterDictListRef)
                
    else:
        # dict is empty, so store complete parameter string
        parameterDictListRef.append(currentDict)