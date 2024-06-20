import logging
import numpy as np
import unittest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from modules import ModelDataSet as mds
from modules import SequenceRepresentation as sr
from modules import sequtils as su
from modules.ProfileFindingSetup import _oneHot

class TestModelDataSite(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=42)
        self.testdatapath = "/home/ebelm/genomegraph/learn_specific_profiles/tests/testdata.json"
        self.genomes = sr.loadJSONGenomeList(self.testdatapath)
        self.rawSeqs_DNA: list[list[list[str]]] = []
        self.rawSeqs_Translated: list[list[list[str]]] = []
        for genome in self.genomes:
            sequences = [[seq.getSequence(), seq.getSequence(rc=True)] for seq in genome]
            self.rawSeqs_DNA.append(sequences)

            translated_sequences = [[sr.TranslatedSequence(seq, 0).getSequence(),
                                     sr.TranslatedSequence(seq, 1).getSequence(),
                                     sr.TranslatedSequence(seq, 2).getSequence(),
                                     sr.TranslatedSequence(seq, 3).getSequence(),
                                     sr.TranslatedSequence(seq, 4).getSequence(),
                                     sr.TranslatedSequence(seq, 5).getSequence()] for seq in genome]
            self.rawSeqs_Translated.append(translated_sequences)


    def tearDown(self):
        pass


    def _sampleKmers(self, rawSeqs: list[list[list[str]]], k: int, n: int):
        """ draw n k-mers randomly from the raw sequences. Returns list[str] of k-mers and 
            list[(g,s,f,start_pos)] of sites """
        nsites = sum([sum([sum([len(fseq) for fseq in seq]) for seq in genome]) for genome in rawSeqs])
        assert nsites >= n*k, f"[TestModelDataSite._sampleKmers] Not enough sites in rawSeqs to draw {n=} {k=}-mers"

        
        forbidden = set() # track already taken sites so kmers don't overlap
        kmers = []
        sites = []
        while len(kmers) < n:
            g = self.rng.choice(len(rawSeqs), 1)[0]       # choose a genome index
            s = self.rng.choice(len(rawSeqs[g]), 1)[0]    # random sequence from that genome
            f = self.rng.choice(len(rawSeqs[g][s]), 1)[0] # random frame from that sequence
            seq = rawSeqs[g][s][f]
            assert len(seq) >= k, f"[TestModelDataSite._sampleKmers] Sequence in {(g,s,f)} too short for {k=}-mer"
            start = self.rng.integers(0, len(seq)-k+1, 1)[0]
            if not any([(g,s,f,start+i) in forbidden for i in range(k)]):
                forbidden.update([(g,s,f,start+i) for i in range(k)])
                kmers.append(seq[start:start+k])
                sites.append((g, s, f, start))
                
        assert len(kmers) == n
        assert len(sites) == n
        return kmers, sites


    def test_DataMode(self):
        self.assertEqual(mds.DataMode.DNA.value, 1)
        self.assertEqual(mds.DataMode.Translated.value, 2)
        self.assertEqual(len(mds.DataMode), 2)


    def test_Alphabet(self):
        self.assertEqual(mds._DNA_ALPHABET, [b for b in "ACGT"])
        self.assertEqual(mds._TRANSLATED_ALPHABET, [a for a in "CKEWTGYAINVHSDFMRLPQ*"])
        self.assertEqual(len(mds._TRANSLATED_ALPHABET), 21)


    def test_backgroundFreqs(self):
        for datamode in mds.DataMode:
            if datamode == mds.DataMode.DNA:
                alphabet = mds._DNA_ALPHABET
                sequences = self.rawSeqs_DNA
            else:
                alphabet = mds._TRANSLATED_ALPHABET
                sequences = self.rawSeqs_Translated

            Q = mds.backgroundFreqs(sequences, alphabet)
            self.assertEqual(Q.shape, (len(alphabet),))
            self.assertAlmostEqual(Q.sum(), 1, places=5) # float imprecisions
            self.assertTrue((Q >= 0).all())
            self.assertTrue((Q <= 1).all())

            # test on a "dumb" sequence where each character appears once, i.e. the alphabet
            Q = mds.backgroundFreqs([[[alphabet]]], alphabet)
            self.assertTrue((Q == 1/len(alphabet)).all())


    def test_batch_creation(self):
        # some assertions about how the training tensor X is created
        def random_seq(n):
            return "".join([self.rng.choice(mds._DNA_ALPHABET) for _ in range(n)])

        # 3 "genomes" with different number of sequences (max. 4) and sequence lengths (max. 555)
        dummydata_dna = [
            [random_seq(250), random_seq(300)],
            [random_seq(420), random_seq(123), random_seq(321), random_seq(333)],
            [random_seq(555)]
        ]

        for datamode in mds.DataMode:
            if datamode == mds.DataMode.DNA:
                alphabet = mds._DNA_ALPHABET
                f = 2
                dummydata = dummydata_dna
            else:
                alphabet = mds._TRANSLATED_ALPHABET
                f = 6
                dummydata = [
                    [sr.TranslatedSequence(sr.Sequence("x","x","+",0,sequence=seq), 0).getSequence() for seq in genome] 
                    for genome in dummydata_dna] # only frame 0, ignore others for now

            # training tensor X fits each sequence (all frames) in a single tile:
            tile_size = len(dummydata[2][0]) # longest seq

            X = np.zeros((4, 3, 2, tile_size, len(alphabet)), dtype=np.float32) # (ntiles, N, frame_dim, ...)
            X[0,0,0,:len(dummydata[0][0]),:] = _oneHot(dummydata[0][0], alphabet)
            X[1,0,0,:len(dummydata[0][1]),:] = _oneHot(dummydata[0][1], alphabet)
            X[0,1,0,:len(dummydata[1][0]),:] = _oneHot(dummydata[1][0], alphabet)
            X[1,1,0,:len(dummydata[1][1]),:] = _oneHot(dummydata[1][1], alphabet)
            X[2,1,0,:len(dummydata[1][2]),:] = _oneHot(dummydata[1][2], alphabet)
            X[3,1,0,:len(dummydata[1][3]),:] = _oneHot(dummydata[1][3], alphabet)
            X[0,2,0,:len(dummydata[2][0]),:] = _oneHot(dummydata[2][0], alphabet)
            
            # data set with tiles big enough to fit all sequences in one tile, and enough tiles to fit all sequences
            # in one X
            genomes = [sr.Genome([sr.Sequence(f"{i}", "chr1", "+", 0, sequence=seq) for seq in dummydata_dna[i]])
                       for i in range(len(dummydata))]
            data = mds.ModelDataSet(genomes, datamode, tile_size=tile_size, tiles_per_X=4, batch_size=1)
            for X_, _ in data.getDataset():
                self.assertEqual(X_.shape, (1, 4, 3, f, tile_size, len(alphabet)))
                X_ = X_[0,:,:,:,:,:] # remove batch dimension
                self.assertFalse((X_.numpy() == 0).all())
                self.assertTrue((X_.numpy()[:,:,0,:,:] == X[:,:,0,:,:]).all()) # ignore reverse complements for now

            # ----------------------------------------------------------------------------------------------------------

            # test if batch data can be assembled back into the original sequences

            tile_size = 10 # shorter tiles to break up sequences
            genomes = [sr.Genome([sr.Sequence(f"{i}", "chr1", "+", 0, sequence=seq) for seq in dummydata_dna[i]])
                       for i in range(len(dummydata))]
            data = mds.ModelDataSet(genomes, datamode, tile_size=tile_size, tiles_per_X=1, batch_size=1)
            restored_data = [[], [], []]
            expected_data = [[], [], []]
            for g in range(3):
                for s in range(f):
                    restored_data[g].append("") # empty sequences for each frame. There is no easy way to determine
                                                #  where the sequences start and end in the tiles, so just concatenate
                                                #  them all and do the same with the original data and see if it matches
                    concatseq = ""
                    for c in range(len(dummydata[g])):
                        if datamode == mds.DataMode.DNA:
                            rawseq = data.training_data.getSequence(g, c, 0).getSequence(rc=(s==1))
                            if s == 1:
                                concatseq = rawseq + concatseq
                            else:
                                concatseq = concatseq + rawseq
                        else:
                            rawseq = data.training_data.getSequence(g, c, s).getSequence()
                            if s >= 3:
                                concatseq = rawseq + concatseq
                            else:
                                concatseq = concatseq + rawseq

                    expected_data[g].append(concatseq)

            for X, _ in data.getDataset():
                self.assertEqual(X.shape, (1, 1, 3, f, tile_size, len(alphabet))) # (b, ntiles, N, frame_dim, ...)
                X = X[0,:,:,:,:,:] # remove batch dimension
                for g in range(3):
                    for s in range(f):
                        tile = X.numpy()[0,g,s,:,:]
                        tileseql = []
                        for i in range(tile_size):
                            self.assertIn(tile[i,:].sum(), [0, 1])
                            if tile[i,:].sum() == 1:
                                cidx = list(tile[i,:]).index(1)
                                tileseql.append(alphabet[cidx])

                        tileseq = "".join(tileseql)
                        if (datamode==mds.DataMode.DNA and s >= 1) or (datamode==mds.DataMode.Translated and s >= 3):
                            restored_data[g][s] = tileseq + restored_data[g][s] # reverse tiling
                        else:
                            restored_data[g][s] = restored_data[g][s] + tileseq
                
            for g in range(3):
                self.assertListEqual(restored_data[g], expected_data[g])


    def test_ModelDataSet_softmasking(self):
        for datamode in mds.DataMode:
            if datamode == mds.DataMode.DNA:
                alphabet = mds._DNA_ALPHABET
                rawSeqs = self.rawSeqs_DNA
            else:
                alphabet = mds._TRANSLATED_ALPHABET
                rawSeqs = self.rawSeqs_Translated

            # training tensor X fits each sequence (all frames) in a single tile:
            tile_size = max([max([max([len(fseq) for fseq in seq]) for seq in genome]) for genome in rawSeqs])
            tiles_per_X = max([len(genome) for genome in rawSeqs]) # one tile per sequence, all sequences in one X
            print(f"tile_size: {tile_size}, tiles_per_X: {tiles_per_X}")

            data = mds.ModelDataSet(self.genomes, datamode, tile_size=tile_size, tiles_per_X=tiles_per_X,
                                    batch_size=1)
            self.assertEqual(data.training_data.datamode, datamode)
            self.assertEqual(data.alphabet, alphabet)
            self.assertEqual(data.alphabet_size(), len(alphabet))
            rawdata = data.getRawData()
            self.assertListEqual(rawdata, rawSeqs)

            # draw a couple of subsequences and check if softmasking is applied correctly
            k = 10
            n = 100
            kmers, sites = self._sampleKmers(rawSeqs, k, n)
            for i in range(n):
                kmer = kmers[i]
                kmer_lower = kmer.lower()
                if '*' in kmer_lower:
                    kmer_lower = kmer_lower.replace('*', ' ')
                site = sites[i]
                kmer_at_site_before_sm = data.softmask(site[0], site[1], site[2], site[3], k)
                self.assertEqual(kmer_at_site_before_sm, kmer)
                softmasked_kmer = data.getRawData()[site[0]][site[1]][site[2]][site[3]:site[3]+k]
                if kmer == kmer.lower(): # catch case where randomly chosen site was already softmasked in the raw data
                    self.assertEqual(softmasked_kmer, kmer)
                else:
                    self.assertNotEqual(softmasked_kmer, kmer)
                    self.assertEqual(softmasked_kmer, kmer_lower)

            # now, training sequences should be softmasked -> check if batches (i.e. one hot encoded training tensors)
            # also reflect that (all-0)
            for X, _ in data.getDataset():
                self.assertEqual(X.shape, (1, 1, len(rawSeqs), data.frame_dimension_size(), tile_size, len(alphabet)))
                for site in sites:
                    g, s, f, start = site
                    self.assertFalse((X.numpy()[0,s,g,f,:,:] == 0).all())
                    self.assertTrue((X.numpy()[0,s,g,f,start:start+k,:] == 0).all())


    def test_ModelDataSet_siteConversion(self):
        # Idea: 
        # input: Sites array with training data coordinates, i.e. g,c,f,tile_start_pos,tile_pos,pIdx with 
        #           f in [0,1] for DNA, f in range(6) for AA
        # output: List of Occurrences with position and strand w.r.t. DNA sr.Sequence objects
        # 
        # expected behaviour: 
        #   The Occurrences' positions should refer to the top strand of the DNA sequence
        #   DNA mode: Input from frame 0 -> Occurrence `occ` should have pos = tile_start_pos+tile_pos, strand = +
        #                 and the extracted (input) k-mer should be the same as occ.sequence[occ.pos:occ.pos+k]
        #             Input from frame 1 -> Occurrence `occ` should have 
        #                 pos = len(sequence) - (tile_start_pos+tile_pos + k) - 1, strand = - and the extracted (input)
        #                 k-mer should be the reverse complement of occ.sequence[occ.pos:occ.pos+k]
        #   Translated mode: Occurrence `occ`s site should be translated into the extracted (input) k-mer
        for datamode in mds.DataMode:
            rawSeqs = self.rawSeqs_DNA if datamode == mds.DataMode.DNA else self.rawSeqs_Translated
            data = mds.ModelDataSet(self.genomes, datamode)
            k = 20
            kmers, sites = self._sampleKmers(rawSeqs, k, 100)
            
            sitearray = np.zeros((len(sites), 6), dtype=np.int32)
            for i, (g,s,f,start_pos) in enumerate(sites):
                tile_start_pos = 0 # TODO: maybe make this random and subtract from start_pos
                profile_idx = 0
                sitearray[i,:] = [g,s,f,tile_start_pos,start_pos,profile_idx]

            # get converted sites from ModelDataSet
            print(f"[DEBUG] >>> sites:\n{sitearray}")
            print(f"[DEBUG] >>> rawseqs: {[[[len(f) for f in seq] for seq in genome] for genome in rawSeqs]}")
            occs = data.convertModelSites(sitearray, k)
            for i, (g,s,f,start_pos) in enumerate(sites):
                print(f"[DEBUG] >>> {i=}, {g=}, {s=}, {f=}, {start_pos=}")
                occ = occs[i]
                kmer = kmers[i]

                if datamode == mds.DataMode.DNA:
                    if f == 0:
                        exp_pos = start_pos
                        exp_strand = '+'
                    else:
                        exp_pos = len(rawSeqs[g][s][f]) - (start_pos + k)
                        exp_strand = '-'
                    self.assertEqual(occ.position, exp_pos)
                    self.assertEqual(occ.strand, exp_strand)
                    extracted_kmer = occ.getSite(k)
                    self.assertEqual(extracted_kmer, kmer)

                if datamode == mds.DataMode.Translated:
                    # raw site points to the right aa-k-mer
                    raw_extraced_kmer = data.training_data.getSequence(g,s,f).getSequence()[start_pos:start_pos+k]
                    self.assertEqual(raw_extraced_kmer, kmer)

                    # do the conversion manually for clarity
                    fwd_dna_seq = data.training_data.getSequence(g,s,f).genomic_sequence.getSequence()
                    if f == 0:
                        tseq = su.sequence_translation(fwd_dna_seq)
                        fwd_dna_pos = start_pos*3
                    elif f == 1:
                        tseq = su.sequence_translation(fwd_dna_seq[1:])
                        fwd_dna_pos = start_pos*3 + 1
                    elif f == 2:
                        tseq = su.sequence_translation(fwd_dna_seq[2:])
                        fwd_dna_pos = start_pos*3 + 2
                    elif f == 3:
                        tseq = su.sequence_translation(fwd_dna_seq, rc=True)
                        rc_dna_pos = (start_pos*3) + k*3 - 1
                        fwd_dna_pos = len(fwd_dna_seq) - rc_dna_pos - 1
                    elif f == 4:
                        tseq = su.sequence_translation(fwd_dna_seq[:-1], rc=True)
                        rc_dna_pos =  (start_pos*3 + 1) + k*3 - 1
                        fwd_dna_pos = len(fwd_dna_seq) - rc_dna_pos - 1
                    elif f == 5:
                        tseq = su.sequence_translation(fwd_dna_seq[:-2], rc=True)
                        rc_dna_pos =  (start_pos*3 + 2) + k*3 - 1
                        fwd_dna_pos = len(fwd_dna_seq) - rc_dna_pos - 1

                    self.assertEqual(tseq, data.training_data.getSequence(g,s,f).getSequence())
                    self.assertEqual(tseq[start_pos:start_pos+k], kmer)

                    fwd_dna_kmer = fwd_dna_seq[fwd_dna_pos:fwd_dna_pos+(3*k)]
                    self.assertEqual(su.sequence_translation(fwd_dna_kmer, (f >= 3)), kmer)

                    # test if Occurrence object has the correct position coming from convertModelSites()
                    self.assertEqual(fwd_dna_pos, occ.position)
                    extracted_kmer = occ.getSite(k*3)
                    extracted_kmer_translated = su.sequence_translation(extracted_kmer)
                    self.assertEqual(extracted_kmer_translated, kmer) # check if the extracted k-mer is correct


    def test_ModelDataSet_siteConversionFromModel(self):
        # same as above, but this time with tiling
        for datamode in mds.DataMode:
            tile_size = 33 # enforce that sequences are broken up in tiles -> test if tile_start is handled correctly
            tiles_per_X = 1
            batch_size = 1
            N = len(self.genomes)
            f = 6 if datamode == mds.DataMode.Translated else 2
            alphabet = mds._TRANSLATED_ALPHABET if datamode == mds.DataMode.Translated else mds._DNA_ALPHABET

            data = mds.ModelDataSet(self.genomes, datamode, tile_size=tile_size, tiles_per_X=tiles_per_X,
                                    batch_size=batch_size)
            k = 10

            # track the tiles and how long the sequences are in them for sampling
            tiles = [] 
            for X, posTrack in data.getDataset(withPosTracking=True):
                self.assertEqual(X.shape, (batch_size, tiles_per_X, N, f, tile_size, len(alphabet)))
                self.assertEqual(posTrack.shape, (batch_size, tiles_per_X, N, f, 4))
                self.assertEqual(X.shape[0], 1)
                self.assertEqual(X.shape[1], 1)

                tilelens = [[0 for _ in range(f)] for _ in range(N)]
                for g in range(N):
                    for s in range(f):
                        if posTrack[0,0,g,s,3] == -1:
                            continue # exhausted sequence tile

                        tilelen = tile_size

                        # from tile end, check how many all-0 rows there are -> length of tile up until there
                        # (this is a bit hacky, but it's a good enough way to determine the length of the sequence in 
                        #  the tile. If a tile ends in ambiguous or softmasked positions, the tile is wrongly assumed
                        #  exhausted but that should be rare enough to not matter)
                        basesum = np.sum(X[0,0,g,s,:,:], axis=1)
                        self.assertEqual(basesum.shape, (tile_size,))
                        i = tile_size - 1
                        while basesum[i] == 0 and i >= 0:
                            i -= 1
                            tilelen -= 1

                        self.assertGreaterEqual(tilelen, 0)
                        self.assertLessEqual(tilelen, tile_size)
                        tilelens[g][s] = tilelen

                tiles.append(tilelens)

            # sample k-mer sites from tiles
            tilesites = []
            while len(tilesites) < 100:
                t = self.rng.choice(len(tiles), 1)[0]         # choose a tile
                g = self.rng.choice(N, 1)[0]                  # choose a genome index
                s = self.rng.choice(f, 1)[0]                  # random frame from that genome tile
                tilelen = tiles[t][g][s]
                if tilelen < k:
                    continue

                start = self.rng.integers(0, tilelen-k+1, 1)[0]
                tilesites.append((t, g, s, start))
                    
            assert len(tilesites) == 100

            # get k-mers from tiles
            kmers = []
            sites = []
            tidx = 0
            for X, posTrack in data.getDataset(withPosTracking=True):
                tsites = [s for s in tilesites if s[0] == tidx]
                for _, g, s, site_start in tsites:
                    self.assertEqual(posTrack[0,0,g,s,0], g)
                    self.assertEqual(posTrack[0,0,g,s,2], s)
                    c = posTrack[0,0,g,s,1]
                    tile_start = posTrack[0,0,g,s,3]

                    tile = X.numpy()[0,0,g,s,:,:]
                    oh_kmer = tile[site_start:site_start+k,:]
                    kmer = []
                    for i in range(k):
                        if oh_kmer[i,:].sum() == 0:
                            kmer.append(' ')
                        else:
                            self.assertEqual(oh_kmer[i,:].sum(), 1)
                            cidx = np.argmax(oh_kmer[i,:])
                            kmer.append(alphabet[cidx])

                    kmer = "".join(kmer)
                    kmers.append(kmer)
                    sites.append((g, c, s, tile_start, site_start, 0))

                tidx += 1
            
            sitearray = np.zeros((len(sites), 6), dtype=np.int32)
            for i, (g,c,s,tile_start,site_start,pIdx) in enumerate(sites):
                sitearray[i,:] = [g,c,s,tile_start,site_start,pIdx]

            # get converted sites from ModelDataSet
            occs = data.convertModelSites(sitearray, k)
            for i, (g,c,s,tile_start,site_start,pIdx) in enumerate(sites):
                occ = occs[i]
                kmer = kmers[i]

                if datamode == mds.DataMode.DNA:
                    extracted_kmer_raw = occ.getSite(k)
                    extracted_kmer = "".join([c if c in alphabet else " " for c in extracted_kmer_raw])
                    self.assertEqual(extracted_kmer, kmer)
                else:
                    extracted_kmer = occ.getSite(k*3)
                    extracted_kmer_translated = su.sequence_translation(extracted_kmer)
                    self.assertEqual(extracted_kmer_translated, kmer)