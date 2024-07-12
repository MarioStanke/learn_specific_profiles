import logging
import numpy as np
import unittest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from modules import position_conversion as pc
from modules import SequenceRepresentation as sr
from modules import utils

class TestModelDataSite(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=42)


    def tearDown(self):
        pass


    def test_fwd_to_rc(self):
        # -- a  b  c  d  e  f  g  h  i  j  --> fwd (len=10) 
        #    0  1  2  3  4  5  6  7  8  9
        #                <==> 
        # -- j' i' h' g' f' e' d' c' b' a' --> rc
        #    0  1  2  3  4  5  6  7  8  9
        self.assertEqual(pc.fwd_to_rc(0, 10), 9)
        self.assertEqual(pc.fwd_to_rc(1, 10), 8)
        self.assertEqual(pc.fwd_to_rc(2, 10), 7)
        self.assertEqual(pc.fwd_to_rc(3, 10), 6)
        self.assertEqual(pc.fwd_to_rc(4, 10), 5)
        self.assertEqual(pc.fwd_to_rc(5, 10), 4)
        self.assertEqual(pc.fwd_to_rc(6, 10), 3)
        self.assertEqual(pc.fwd_to_rc(7, 10), 2)
        self.assertEqual(pc.fwd_to_rc(8, 10), 1)
        self.assertEqual(pc.fwd_to_rc(9, 10), 0)

        self.assertRaises(AssertionError, pc.fwd_to_rc, 10, 10) # out of range
        self.assertRaises(AssertionError, pc.fwd_to_rc, -1, 10)

        # test on random Sequence objects
        for _ in range(1000):
            seqlen = 1000
            seqstr = "N" * seqlen
            a = self.rng.choice(range(seqlen))
            assert a in range(seqlen)
            seqstr = seqstr[:a] + "A" + seqstr[a+1:] # replace single N-base with A
            seq = sr.Sequence("testspecies", "testchr", "+", 0, sequence=seqstr)
            
            a_rc = pc.fwd_to_rc(a, seqlen)
            seqstr_rc = seq.getSequence(rc=True)
            self.assertEqual(seqstr[a], "A")
            self.assertEqual(seqstr_rc[a_rc], "T")
            self.assertEqual(seqstr_rc.index("T"), a_rc)

            # test that fwd_to_rc is its own inverse
            self.assertEqual(pc.fwd_to_rc(a_rc, seqlen), a)
            seq_rc = sr.Sequence("testspecies", "testchr", "-", 0, sequence=seqstr_rc)
            self.assertEqual(seq_rc.getSequence(rc=True), seqstr)

    
    def test_dna_to_aa(self):
        # test on random sequences
        seqlen = 1000
        seqstr = "".join(self.rng.choice(["G", "C", "T", "A"]) for _ in range(seqlen))
        assert len(seqstr) == seqlen
        seq = sr.Sequence("testspecies", "testchr", "+", 0, sequence=seqstr)
        seqstr_rc = seq.getSequence(rc=True)
        seq_rc = sr.Sequence("testspecies", "testchr", "-", 0, sequence=seqstr_rc)

        # fwd translation
        for i in range(0, seqlen-2):
            frame = i % 3
            seq_trans = sr.TranslatedSequence(seq, frame)
            seqstr_trans = seq_trans.getSequence()

            triplet = seqstr[i:i+3]
            assert triplet in utils._genetic_code
            aa = utils._genetic_code[triplet]

            # dna -> aa pos
            f, aa_pos = pc.dna_to_aa(i)
            self.assertEqual(f, frame)
            self.assertEqual(seqstr_trans[aa_pos], aa)

            # aa -> dna pos
            dna_pos = pc.aa_to_dna(frame, aa_pos)
            self.assertEqual(dna_pos, i)

        # rc translation
        for i in range(0, seqlen-2):
            frame = i % 3
            seq_rc_trans = sr.TranslatedSequence(seq_rc, frame)
            seqstr_rc_trans = seq_rc_trans.getSequence()

            triplet = seqstr_rc[i:i+3]
            assert triplet in utils._genetic_code
            aa = utils._genetic_code[triplet]

            # dna -> aa pos
            f, aa_pos = pc.dna_to_aa(i)
            self.assertEqual(f, frame)
            self.assertEqual(seqstr_rc_trans[aa_pos], aa)

            # aa -> dna pos
            dna_pos = pc.aa_to_dna(frame, aa_pos)
            self.assertEqual(dna_pos, i)


    def test_dna_range_to_aa(self):
        pass # not used yet


    def test_translated_training_case(self):
        # test the entire position conversion pipeline for translated training cases
        fwd_seq = "NNNNNNTTTCAACAATGTGCACGANNNNNNNNNNNN"
        seq = sr.Sequence("testspecies", "testchr", "+", 0, sequence=fwd_seq)
        self.assertEqual(seq.getSequence(), fwd_seq)
        self.assertEqual(seq.getSequence(rc=True), "NNNNNNNNNNNNTCGTGCACATTGTTGAAANNNNNN")
        t0_seq = sr.TranslatedSequence(seq, 0)
        t1_seq = sr.TranslatedSequence(seq, 1)
        t2_seq = sr.TranslatedSequence(seq, 2)
        t3_seq = sr.TranslatedSequence(seq, 3)
        t4_seq = sr.TranslatedSequence(seq, 4)
        t5_seq = sr.TranslatedSequence(seq, 5)
        self.assertEqual(t0_seq.getSequence(), "  FQQCAR    ")
        self.assertEqual(t1_seq.getSequence(), "  FNNVH    ")
        self.assertEqual(t2_seq.getSequence(), "  STMCT    ")
        self.assertEqual(t3_seq.getSequence(), "    SCTLLK  ")
        self.assertEqual(t4_seq.getSequence(), "    RAHC*  ")
        self.assertEqual(t5_seq.getSequence(), "    VHIVE  ")

        # test dna_to_aa and back
        a = 6
        k_dna = 18
        self.assertEqual(fwd_seq[a:a+k_dna], "TTTCAACAATGTGCACGA")

        a_rc = pc.fwd_to_rc(a+k_dna-1, len(fwd_seq))
        self.assertEqual(a_rc, 12)
        self.assertEqual(seq.getSequence(rc=True)[ a_rc : a_rc+k_dna ], "TCGTGCACATTGTTGAAA")

        f0, a_f0 = pc.dna_to_aa(a)
        self.assertEqual(f0, 0)
        self.assertEqual(a_f0, 2)
        self.assertEqual(t0_seq.getSequence()[a_f0:a_f0+6], "FQQCAR")
        self.assertEqual(pc.aa_to_dna(f0, a_f0), a)
        self.assertEqual(utils.sequence_translation(fwd_seq[pc.aa_to_dna(f0, a_f0) : pc.aa_to_dna(f0, a_f0) + k_dna]),
                         "FQQCAR")

        f1, a_f1 = pc.dna_to_aa(a+1)
        self.assertEqual(f1, 1)
        self.assertEqual(a_f1, 2)
        self.assertEqual(t1_seq.getSequence()[a_f1:a_f1+5], "FNNVH")
        self.assertEqual(pc.aa_to_dna(f1, a_f1), a+1)
        self.assertEqual(utils.sequence_translation(fwd_seq[pc.aa_to_dna(f1, a_f1) : pc.aa_to_dna(f1, a_f1) + k_dna-1]),
                         "FNNVH")

        f2, a_f2 = pc.dna_to_aa(a+2)
        self.assertEqual(f2, 2)
        self.assertEqual(a_f2, 2)
        self.assertEqual(t2_seq.getSequence()[a_f2:a_f2+5], "STMCT")
        self.assertEqual(pc.aa_to_dna(f2, a_f2), a+2)
        self.assertEqual(utils.sequence_translation(fwd_seq[pc.aa_to_dna(f2, a_f2) : pc.aa_to_dna(f2, a_f2) + k_dna-2]),
                         "STMCT")

        f3, a_f3 = pc.dna_to_aa(a_rc)
        k_f3 = 6
        self.assertEqual(f3, 0)
        self.assertEqual(a_f3, 4)
        self.assertEqual(t3_seq.getSequence()[a_f3:a_f3+k_f3], "SCTLLK")
        self.assertEqual(pc.aa_to_dna(f3, a_f3), a_rc)
        # aa_kmer:      t3_seq[  4 :  4+6]     = "SCTLLK"
        # dna_rc_kmer:  rc_seq[ 12 : 12+(3*6)] = "TCGTGCACATTGTTGAAA" | pc.aa_to_dna(0, 4) = 12
        # dna_fwd_kmer: fwd_seq[ 6 :  6+(3*6)] = "TTTCAACAATGTGCACGA" | pc.rc_to_fwd(12 + 18 - 1, len(fwd_seq)) = 6
        self.assertEqual(pc.aa_to_dna(f3, a_f3), a_rc)
        self.assertEqual(pc.aa_to_dna(f3, a_f3), 12)
        self.assertEqual(pc.aa_to_dna(f3, a_f3+k_f3), a_rc+(3*k_f3))
        self.assertEqual(pc.aa_to_dna(f3, a_f3+k_f3), 30)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f3, a_f3), len(fwd_seq)), a+k_dna-1)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f3, a_f3), len(fwd_seq)), 23)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f3, a_f3+k_f3)-1, len(fwd_seq)), a)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f3, a_f3+k_f3)-1, len(fwd_seq)), 6)
        self.assertEqual(utils.sequence_translation(
                            fwd_seq[pc.rc_to_fwd(pc.aa_to_dna(f3, a_f3+k_f3)-1, len(fwd_seq)) 
                                    : pc.rc_to_fwd(pc.aa_to_dna(f3, a_f3), len(fwd_seq))+1], rc=True),
                         "SCTLLK")
        

        f4, a_f4 = pc.dna_to_aa(a_rc+1)
        k_f4 = 5
        self.assertEqual(f4, 1)
        self.assertEqual(a_f4, 4)
        self.assertEqual(t4_seq.getSequence()[a_f4:a_f4+k_f4], "RAHC*")
        self.assertEqual(pc.aa_to_dna(f4, a_f4), a_rc+1)
        # aa_kmer:      t4_seq[  4 :  4+5]     = "RAHC*"
        # dna_rc_kmer:  rc_seq[ 13 : 13+(3*5)] = "CGTGCACATTGTTGA" | pc.aa_to_dna(1, 4) = 13
        # dna_fwd_kmer: fwd_seq[ 8 :  8+(3*5)] = "TCAACAATGTGCACG" | pc.rc_to_fwd(13 + 15 - 1, len(fwd_seq)) = 6
        self.assertEqual(pc.aa_to_dna(f4, a_f4), a_rc+1)
        self.assertEqual(pc.aa_to_dna(f4, a_f4), 13)
        self.assertEqual(pc.aa_to_dna(f4, a_f4+k_f4), a_rc+1+(3*k_f4))
        self.assertEqual(pc.aa_to_dna(f4, a_f4+k_f4), 28)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f4, a_f4), len(fwd_seq)), a+2+15-1)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f4, a_f4), len(fwd_seq)), 22)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f4, a_f4+k_f4)-1, len(fwd_seq)), a+2)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f4, a_f4+k_f4)-1, len(fwd_seq)), 8)
        self.assertEqual(utils.sequence_translation(
                            fwd_seq[pc.rc_to_fwd(pc.aa_to_dna(f4, a_f4+k_f4)-1, len(fwd_seq)) 
                                    : pc.rc_to_fwd(pc.aa_to_dna(f4, a_f4), len(fwd_seq))+1], rc=True),
                         "RAHC*")

        f5, a_f5 = pc.dna_to_aa(a_rc+2)
        k_f5 = 5
        self.assertEqual(f5, 2)
        self.assertEqual(a_f5, 4)
        self.assertEqual(t5_seq.getSequence()[a_f5:a_f5+k_f5], "VHIVE")
        self.assertEqual(pc.aa_to_dna(f5, a_f5), a_rc+2)
        # aa_kmer:      t5_seq[  4 :  4+5]     = "VHIVE"
        # dna_rc_kmer:  rc_seq[ 14 : 14+(3*5)] = "GTGCACATTGTTGAA" | pc.aa_to_dna(1, 4) = 13
        # dna_fwd_kmer: fwd_seq[ 7 :  7+(3*5)] = "TTCAACAATGTGCAC" | pc.rc_to_fwd(13 + 15 - 1, len(fwd_seq)) = 6
        self.assertEqual(pc.aa_to_dna(f5, a_f5), a_rc+2)
        self.assertEqual(pc.aa_to_dna(f5, a_f5), 14)
        self.assertEqual(pc.aa_to_dna(f5, a_f5+k_f5), a_rc+2+(3*k_f5))
        self.assertEqual(pc.aa_to_dna(f5, a_f5+k_f5), 29)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f5, a_f5), len(fwd_seq)), a+1+15-1)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f5, a_f5), len(fwd_seq)), 21)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f5, a_f5+k_f5)-1, len(fwd_seq)), a+1)
        self.assertEqual(pc.rc_to_fwd(pc.aa_to_dna(f5, a_f5+k_f5)-1, len(fwd_seq)), 7)
        self.assertEqual(utils.sequence_translation(
                            fwd_seq[pc.rc_to_fwd(pc.aa_to_dna(f5, a_f5+k_f5)-1, len(fwd_seq)) 
                                    : pc.rc_to_fwd(pc.aa_to_dna(f5, a_f5), len(fwd_seq))+1], rc=True),
                         "VHIVE")