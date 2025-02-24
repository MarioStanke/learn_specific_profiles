import numpy as np
from tqdm import tqdm
import unittest

from modules import dataset as ds
from modules import sequtils as su

class TestDataset(unittest.TestCase):
    def test_backGroundAAFreqs(self):
        self.skipTest("Not implemented")

    def test_translateSequenceTiles(self):
        self.skipTest("Not implemented")

    def test_oneHotAndToAA(self):
        aaSeq = ''.join([su.genetic_code[c] for c in su.genetic_code.keys()])
        aaSeq += ' '
        self.assertEqual(aaSeq, 
                         "IIIMTTTTNNKKSSRRLLLLPPPPHHQQRRRRVVVVAAAADDEEGGGGSSSSFFLLYY**CC*W ")
        oh = np.array(
            # C K E W T G Y A I N V H S D F M R L P Q *
            [[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], # I
             [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], # I
             [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], # I
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], # M
             [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # T
             [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # T
             [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # T
             [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # T
             [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], # N
             [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], # N
             [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # K
             [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # K
             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], # S
             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], # S
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], # R
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], # R
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], # L
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], # L
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], # L
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], # L
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], # P
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], # P
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], # P
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], # P
             [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], # H
             [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], # H
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], # Q
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], # Q
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], # R
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], # R
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], # R
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], # R
             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], # V
             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], # V
             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], # V
             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], # V
             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # A
             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # A
             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # A
             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # A
             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], # D
             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], # D
             [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # E
             [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # E
             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # G
             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # G
             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # G
             [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # G
             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], # S
             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], # S
             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], # S
             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], # S
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], # F
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], # F
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], # L
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], # L
             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Y
             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Y
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], # *
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], # *
             [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # C
             [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # C
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], # *
             [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # W
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # ' '
            ], dtype=np.float32
        )
        dsOH = ds.oneHot(aaSeq)
        self.assertEqual(dsOH.shape, oh.shape)
        self.assertTrue(np.all(dsOH == oh), f"One hot encoding failed:\n{dsOH}\n{oh}")
        self.assertEqual(ds.to_aa(oh, dropEmptyColumns=False), aaSeq)
        self.assertEqual(ds.to_aa(oh, dropEmptyColumns=True), aaSeq[:-1])

    def test_seqlistFromGenomes(self):
        genomes = ds.simulateGenomes(8, 6250, 100)
        seqlist = ds.seqlistFromGenomes(genomes)
        self.assertEqual(len(seqlist), len(genomes))
        for i in range(len(genomes)):
            with self.subTest(i=i):
                self.assertEqual(len(seqlist[i]), len(genomes[i]))
                for j in range(len(genomes[i])):
                    with self.subTest(j=j):
                        self.assertEqual(seqlist[i][j], genomes[i][j].getSequence())

    def test_restoreGenomePosition(self):
        seqlen = [420, 421, 422]
        tilesize = 99 # has to be mod3 == 0
        ks = [0, 1, 2, 3, 4, 5]
        for sl in seqlen:
            for aak in ks:
                k = aak * 3
                for ts in [0,1,2]:
                    tilestart = ts
                    aai = 0
                    for dnai in range(ts, sl, 3):
                        if dnai >= tilestart + tilesize:
                            tilestart += tilesize
                            aai = 0

                        self.assertEqual(ds.restoreGenomePosition(aai, tilestart, aak, fwd=True), dnai, 
                                            f"sl={sl}, k={k}, ts={ts}, dnai={dnai}, tilestart={tilestart}, aai={aai}")
                        aai += 1

                    aai = 0
                    tilestart = sl - 1
                    for dnai in range(sl-3, -1, -3):
                        if dnai <= tilestart - tilesize:
                            tilestart -= tilesize
                            aai = 0

                        self.assertEqual(ds.restoreGenomePosition(aai, tilestart, aak, fwd=False), dnai-(k-3), 
                                         f"sl={sl}, aak={aak}, k={k}, ts={ts}, dnai={dnai}, tilestart={tilestart}," \
                                                                                                         + f"aai={aai}")
                        aai += 1


    def test_Generator(self):
        #self.skipTest("Takes very long")
        # copy of test functions in dataset.py
        genomes = ds.seqlistFromGenomes(ds.simulateGenomes(8, 6250, 100))
        tile_size = 1000 // 3
        ntiles = 7
        limit = 50000

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
                aa_seqs = su.three_frame_translation(testgenome[g][i])
                aa_seqs.extend(su.three_frame_translation(testgenome[g][i], True))
                for f in range(len(aa_seqs)):
                    genome_aa[g][f] += aa_seqs[f].replace(' ', '')

        # for each genome and each frame, concatenate translated aa seqs
        X_to_genome_aa = [[""]*6 for _ in range(len(testgenome))]

        # create generator
        Xgen = ds.createBatch(ntiles, tile_size, testgenome)

        # iterate through generator, transforming and concatenating aa sequences
        # X.shape [ntiles, N, 6, tile_size, su.aa_alphabet_size]
        for X, _ in tqdm(Xgen):
            for t in range(X.shape[0]):
                for g in range(X.shape[1]):
                    for f in range(X.shape[2]):
                        x_aa_seq = ds.to_aa(X[t,g,f,:,:])
                        X_to_genome_aa[g][f] += x_aa_seq

        # compare aa sequences
        if np.all(genome_aa == X_to_genome_aa):
            self.assertTrue(np.all(genome_aa == X_to_genome_aa), str(genome_aa)+"\n\n!=\n\n"+str(X_to_genome_aa))
            print("[INFO] >>> testGenerator - restore sequences: All good")
            
        # In case of failure, see where the sequences differ or if only frames were shifted (for some unknown reason)
        else:
            self.assertEqual(len(genome_aa), len(X_to_genome_aa))
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

                            self.assertTrue(False, "Abort")
                            
            print("[INFO] >>> testGenerator - restore sequences: rc frames shifted, otherwise all good")

        # test position restoring
        XgenPos = ds.createBatch(ntiles, tile_size, testgenome, True)
        k = 11
        for X, P in tqdm(XgenPos): # (tilesPerX, N, 6, T, 21), 
                                   # (tilesPerX, N, 6, (genomeIDs, contigIDs, startPos, aa_seqlen))
            for t in range(X.shape[0]):
                for g in range(X.shape[1]):
                    for f in range(X.shape[2]):
                        for p in range(X.shape[3]-k+1):
                            if P[t,g,f,0] != -1:
                                self.assertEqual(P[t,g,f,0], g, str(P[t,g,f,:]))
                                c = P[t,g,f,1]
                                kmerOH = X[t,g,f,p:(p+k),:]
                                kmerAA = ds.to_aa(kmerOH, False)
                                self.assertEqual(len(kmerAA), k, 
                                                 str(k)+", '"+str(kmerAA)+"' ("+str(len(kmerAA))+")\n"+str(kmerOH))
                                pos = ds.restoreGenomePosition(p, P[t,g,f,2], k, (f<3))
                                end = pos+(k*3)
                                # sometimes negative for rc frames or reaching over the sequence for fwd frames, skip
                                if pos >= 0 and end <= len(testgenome[g][c]):
                                    sourceKmer = testgenome[g][c][pos:end]
                                    sourceKmerAA = su.sequence_translation(sourceKmer) \
                                        if f < 3 else su.sequence_translation(sourceKmer, True)
                                    self.assertEqual(len(kmerAA), len(sourceKmerAA), 
                                                     "\n'"+sourceKmerAA+"' !=\n'"+kmerAA+"'\n" \
                                                        +str((g,c,t,f,p,pos,P[t,g,f,:])))
                                    self.assertEqual(kmerAA, sourceKmerAA, 
                                                     "\n'"+sourceKmerAA+"' !=\n'"+kmerAA+"'\n" \
                                                        +str((g,c,t,f,p,pos,P[t,g,f,:])))

        print("[INFO] >>> testGenerator - restore positions: All good")

    def test_getForbiddenPositions(self):
        self.assertEqual(ds.getForbiddenPositions([], k=3, slen=10), set([]))
        self.assertEqual(ds.getForbiddenPositions([0], k=3, slen=10), set([0,1,2]))
        self.assertEqual(ds.getForbiddenPositions([1], k=3, slen=10), set([0,1,2,3]))
        self.assertEqual(ds.getForbiddenPositions([2], k=3, slen=10), set([0,1,2,3,4]))
        self.assertEqual(ds.getForbiddenPositions([3], k=3, slen=10), set([1,2,3,4,5]))
        self.assertEqual(ds.getForbiddenPositions([4], k=3, slen=10), set([2,3,4,5,6]))
        self.assertEqual(ds.getForbiddenPositions([5], k=3, slen=10), set([3,4,5,6,7]))
        self.assertEqual(ds.getForbiddenPositions([6], k=3, slen=10), set([4,5,6,7,8]))
        self.assertEqual(ds.getForbiddenPositions([7], k=3, slen=10), set([5,6,7,8,9]))
        self.assertEqual(ds.getForbiddenPositions([8], k=3, slen=10), set([6,7,8,9]))
        self.assertEqual(ds.getForbiddenPositions([9], k=3, slen=10), set([7,8,9]))
        self.assertEqual(ds.getForbiddenPositions([0,1], k=3, slen=10), set([0,1,2,3]))
        self.assertEqual(ds.getForbiddenPositions([0,1,2], k=3, slen=10), set([0,1,2,3,4]))
        self.assertEqual(ds.getForbiddenPositions([0,1,2,3], k=3, slen=10), set([0,1,2,3,4,5]))
        self.assertEqual(ds.getForbiddenPositions([0,1,2,3,4], k=3, slen=10), set([0,1,2,3,4,5,6]))
        self.assertEqual(ds.getForbiddenPositions([0,1,2,3,4,5], k=3, slen=10), set([0,1,2,3,4,5,6,7]))
        self.assertEqual(ds.getForbiddenPositions([0,1,2,3,4,5,6], k=3, slen=10), set([0,1,2,3,4,5,6,7,8]))
        self.assertEqual(ds.getForbiddenPositions([0,1,2,3,4,5,6,7], k=3, slen=10), set([0,1,2,3,4,5,6,7,8,9]))
        self.assertEqual(ds.getForbiddenPositions([0,1,2,3,4,5,6,7,8], k=3, slen=10), set([0,1,2,3,4,5,6,7,8,9]))
        self.assertEqual(ds.getForbiddenPositions([0,1,2,3,4,5,6,7,8,9], k=3, slen=10), set([0,1,2,3,4,5,6,7,8,9]))
        self.assertEqual(ds.getForbiddenPositions([5], k=0, slen=10), set([]))
        self.assertEqual(ds.getForbiddenPositions([5], k=1, slen=10), set([5]))
        self.assertEqual(ds.getForbiddenPositions([5], k=2, slen=10), set([4,5,6]))
        self.assertEqual(ds.getForbiddenPositions([5], k=10, slen=10), set([0,1,2,3,4,5,6,7,8,9]))
        
    def test_insertPatternsToGenomes(self):
        # Copied tests from toy-data branch in jupyter notebook
        
        N = 8            # number of genomes
        repeatsPerGenome = 10 # number of times to insert repeats into each genome
        multiplyRepeats = 1   # multiply repeat patterns
        genome_sizes = [[10000]] * N
                        # in nucleotides
        insertPatterns = ["ATGGCAAGAATTCAATCTACTGCAAATAAAGAA"] 
        repeatPatterns = ['AGAGAACCTGAAGCTACTGCTGAACCTGAAAGA']
        genomes = ds.getRandomGenomes(N, genome_sizes, insertPatterns, repeatPatterns,
                                      mutationProb=0.0, 
                                      multiplyRepeat=[multiplyRepeats],
                                      multipleRepeatInserts=[repeatsPerGenome],
                                      verbose=False)
        
        # assert patterns are all inserted
        self.assertEqual(len(genomes), N, str(len(genomes)))
        for g in range(len(genomes)):
            self.assertEqual(len(genomes[g]), len(genome_sizes[g]), str(len(genomes[g]))+", "+str(g))
            for c in genomes[g]:
                self.assertTrue(c.hasElements(), str(c))
                self.assertEqual(len(c.genomic_elements), 11)
                self.assertEqual(c.genomic_elements[-1].sequence, insertPatterns[0])
                for i in range(0,10):
                    with self.subTest(i=i):
                        self.assertEqual(c.genomic_elements[i].sequence, repeatPatterns[0]*multiplyRepeats)

    def test_getRandomGenomes(self):
        self.skipTest("Not implemented")

    def test_simulateGenomes(self):
        self.skipTest("Not implemented")



if __name__ == '__main__':
    unittest.main(verbosity=2)