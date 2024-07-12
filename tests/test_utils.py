from Bio.Seq import Seq
import unittest

from modules import utils

class TestSeqUtils(unittest.TestCase):
    def test_globals(self):
        self.assertEqual(len(utils._dna_alphabet), 22)
        self.assertEqual(len(utils._complements), 22)
        self.assertEqual(utils._codon_len, 3)
        self.assertEqual(len(utils._genetic_code), 4*16)


    def test_sequence_translation(self):
        self.assertEqual(utils.sequence_translation(""), "")
        self.assertEqual(utils.sequence_translation("A"), "")
        self.assertEqual(utils.sequence_translation("AC"), "")
        self.assertEqual(utils.sequence_translation("ACG"), "T")
        self.assertEqual(utils.sequence_translation("ACG", True), "R")
        self.assertEqual(utils.sequence_translation("ACGT"), "T")
        self.assertEqual(utils.sequence_translation("ACGT", True), "T")
        self.assertEqual(utils.sequence_translation("ACGTA"), "T")
        self.assertEqual(utils.sequence_translation("ACGTA", True), "Y")
        self.assertEqual(utils.sequence_translation("ACGTAC"), "TY")
        self.assertEqual(utils.sequence_translation("ACGTAC", True), "VR")


    def test_six_frame_translation(self):
        # empty sequence
        self.assertEqual(utils.six_frame_translation(''), ['', '', '', '', '', ''])
        # single base
        self.assertEqual(utils.six_frame_translation('A'), ['', '', '', '', '', ''])
        # two bases
        self.assertEqual(utils.six_frame_translation('AT'), ['', '', '', '', '', ''])
        # single codon
        self.assertEqual(utils.six_frame_translation('ATG'), ['M', '', '', 'H', '', ''])
        # 5 bases
        self.assertEqual(utils.six_frame_translation('ATGAA'), ['M', '*', 'E', 'F', 'S', 'H'])
        # 6 bases with N
        self.assertEqual(utils.six_frame_translation('ATGANN'), ['M ', '*', ' ', ' H', ' ', 'S'])

        # all codons
        codons = [cod for cod in utils._genetic_code.keys()]
        s = ''.join(codons)
        f1 = ''.join([utils._genetic_code[cod] for cod in codons])

        s2 = s[1:-2]
        self.assertEqual(len(s2) % 3, 0)
        codons2 = [s2[i:i+3] for i in range(0, len(s2), 3)]
        f2 = ''.join([utils._genetic_code[cod] for cod in codons2])

        s3 = s[2:-1]
        self.assertEqual(len(s3) % 3, 0)
        codons3 = [s3[i:i+3] for i in range(0, len(s3), 3)]
        f3 = ''.join([utils._genetic_code[cod] for cod in codons3])

        s4 = Seq(s).reverse_complement()
        self.assertEqual(len(s4) % 3, 0)
        codons4 = [s4[i:i+3] for i in range(0, len(s4), 3)]
        f4 = ''.join([utils._genetic_code[cod] for cod in codons4])

        s5 = s4[1:-2]
        self.assertEqual(len(s5) % 3, 0)
        codons5 = [s5[i:i+3] for i in range(0, len(s5), 3)]
        f5 = ''.join([utils._genetic_code[cod] for cod in codons5])

        s6 = s4[2:-1]
        self.assertEqual(len(s6) % 3, 0)
        codons6 = [s6[i:i+3] for i in range(0, len(s6), 3)]
        f6 = ''.join([utils._genetic_code[cod] for cod in codons6])

        self.assertEqual(utils.six_frame_translation(s), [f1, f2, f3, f4, f5, f6])


if __name__ == '__main__':
    unittest.main(verbosity=2)