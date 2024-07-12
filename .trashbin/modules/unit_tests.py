""" Simple script to write and run 'unit tests' for the modules as needed """

import sequtils as su

def test_sequtils_convert_six_frame_position():
    # some random DNA sequence
    dna = "CACAACCTACGTATGGTTCTACCTTAAATAATTGAGTGTTAGCCAAGTATCGGTTGCGCGCTTTAGCTCTTAAGTAAACCGATTGAGGCT"
    assert len(dna) == 90 # documentation purpose: sequence is of length 90
    
    # do a manual siz-frame translation and map positions to expected positions
    dna_to_aa = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}} # frame to pos-mapping
    aa_to_dna = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    manual_sft = ['']*6
    for i in range(0, len(dna), 3):
        for f in [0,1,2]:
            if (i+f+3) <= len(dna):
                codon = dna[(i+f):(i+f+3)]
                aa = su.genetic_code[codon] if codon in su.genetic_code else ' '
                dna_to_aa[f][i+f] = len(manual_sft[f])
                dna_to_aa[f][i+f+1] = len(manual_sft[f])
                dna_to_aa[f][i+f+2] = len(manual_sft[f])
                manual_sft[f] += aa
    
    for i in range(len(dna)-1, -1, -3):
        for f in [3,4,5]:
            rf = f-3
            if (i-rf-2) >= 0:
                codon = dna[(i-rf-2):(i+1-rf)]
                codon = codon[::-1].translate(su.rctbl)
                aa = su.genetic_code[codon] if codon in su.genetic_code else ' '
                dna_to_aa[f][i-rf] = len(manual_sft[f])
                dna_to_aa[f][i-rf-1] = len(manual_sft[f])
                dna_to_aa[f][i-rf-2] = len(manual_sft[f])
                manual_sft[f] += aa
    
    sft = su.six_frame_translation(dna)
    assert len(sft) == len(manual_sft), "[ERROR] >>> len(sft) "+str(len(sft))+" != len(manual_sft) "+\
                                        str(len(manual_sft))
    for i in range(len(sft)):
        assert sft[i] == manual_sft[i], "[ERROR] >>> manual_sft "+str(i)+" differs from sft:\n'"+manual_sft[i]+"'\n'"+\
                                        sft[i]+"'"
        
    for f in dna_to_aa:
        for pos in dna_to_aa[f]:
            #if pos < len(dna): # for security
            convPos = su.convert_six_frame_position(pos, f, len(dna), dna_to_aa=True)
            assert convPos == dna_to_aa[f][pos], "[ERROR] >>> dna_to_aa frame "+str(f)+" pos "+str(pos)+\
                                                 " expected "+str(dna_to_aa[f][pos])+" but got "+str(convPos)
                
    # established that dna -> aa is correct, now simply test reverse case
    for f in dna_to_aa:
        for pos in dna_to_aa[f]:
            aapos = dna_to_aa[f][pos]
            if aapos not in aa_to_dna[f]:
                aa_to_dna[f][aapos] = pos
                
            # map to first codon base w.r.t. the forward strand
            aa_to_dna[f][aapos] = min(pos, aa_to_dna[f][aapos])
            
    for f in aa_to_dna:
        for pos in aa_to_dna[f]:
            #if pos < len(sft[f]): # for security
            convPos = su.convert_six_frame_position(pos, f, len(dna), dna_to_aa=False)
            assert convPos == aa_to_dna[f][pos], "[ERROR] >>> aa_to_dna frame "+str(f)+" pos "+str(pos)+\
                                                 " expected "+str(aa_to_dna[f][pos])+" but got "+str(convPos)
                
    return



# =====================================================================================================================



test_sequtils_convert_six_frame_position()