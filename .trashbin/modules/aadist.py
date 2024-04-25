import sequtils as su

def getBackgroundDist(hmmer=False, alphabet=su.aa_alphabet[1:]):
    # Das ist der Erwartungswert einer Dirichletverteilung, die ich auf allen 
    #   Pfam HMMs trainiert habe. Das benutze ich in meinem Code. (Felix)
    dist = [0.08323683, 0.05253151, 0.04916192, 0.04657839, 0.02241498,
            0.05126856, 0.06325443, 0.04734022, 0.03321823, 0.05307856,
            0.07356835, 0.06407219, 0.03507021, 0.03604704, 0.03456395,
            0.07165699, 0.06525507, 0.01756833, 0.03383222, 0.06628203]
    
    if hmmer:
        # Alternativ benutzt HMMER diese Hintergrundverteilung für Insertionen:
        dist = [0.06814074, 0.05513283, 0.05483262, 0.06233763, 0.01200719,
                0.04152498, 0.06513912, 0.09025376, 0.02411455, 0.0371222 ,
                0.06764039, 0.06874096, 0.01430852, 0.0313187 , 0.0647391 ,
                0.09265522, 0.06233763, 0.0102061 , 0.02691612, 0.05053041]
        
    order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
             'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    aadict = dict(zip(order, dist))
    Q = []
    for aa in alphabet:
        if aa in aadict:
            Q.append(aadict[aa])
        else:
            Q.append(1/len(alphabet)) # this messes up the distribution as it no longer sums to 1.0
            
    return Q
    