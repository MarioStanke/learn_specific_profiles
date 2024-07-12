from Bio.Seq import Seq
import unittest

from modules import sequtils as su

class TestSeqUtils(unittest.TestCase):
    def test_globals(self):
        self.assertEqual(su.dna_alphabet_size, 22)
        self.assertEqual(su.codon_len, 3)
        self.assertEqual(su.codon_alphabet_size, 10648)
        self.assertEqual(len(su.genetic_code), 4*16)
        self.assertEqual(su.aa_alphabet_size, 21)
        self.assertEqual(len(su.aa_alphabet), 22)

    def test_sequence_translation(self):
        self.assertEqual(su.sequence_translation(""), "")
        self.assertEqual(su.sequence_translation("A"), "")
        self.assertEqual(su.sequence_translation("AC"), "")
        self.assertEqual(su.sequence_translation("ACG"), "T")
        self.assertEqual(su.sequence_translation("ACG", True), "R")
        self.assertEqual(su.sequence_translation("ACGT"), "T")
        self.assertEqual(su.sequence_translation("ACGT", True), "T")
        self.assertEqual(su.sequence_translation("ACGTA"), "T")
        self.assertEqual(su.sequence_translation("ACGTA", True), "Y")
        self.assertEqual(su.sequence_translation("ACGTAC"), "TY")
        self.assertEqual(su.sequence_translation("ACGTAC", True), "VR")

    def test_three_frame_translation(self):
        sequence = ''.join([c for c in su.genetic_code.keys()])
        self.assertEqual(su.three_frame_translation(sequence), su.six_frame_translation(sequence)[:3])
        self.assertEqual(su.three_frame_translation(sequence, rc=True), su.six_frame_translation(sequence)[3:])
        self.assertEqual(su.three_frame_translation(sequence, offsets=range(1,4)), 
                         su.six_frame_translation(sequence[1:])[:3])
        self.assertEqual(su.three_frame_translation(sequence, rc=True, offsets=range(1,4)), 
                         su.six_frame_translation(sequence[:-1])[3:])
        
    def test_six_frame_translation(self):
        # empty sequence
        self.assertEqual(su.six_frame_translation(''), ['', '', '', '', '', ''])
        # single base
        self.assertEqual(su.six_frame_translation('A'), ['', '', '', '', '', ''])
        # two bases
        self.assertEqual(su.six_frame_translation('AT'), ['', '', '', '', '', ''])
        # single codon
        self.assertEqual(su.six_frame_translation('ATG'), ['M', '', '', 'H', '', ''])
        # 5 bases
        self.assertEqual(su.six_frame_translation('ATGAA'), ['M', '*', 'E', 'F', 'S', 'H'])
        # 6 bases with N
        self.assertEqual(su.six_frame_translation('ATGANN'), ['M ', '*', ' ', ' H', ' ', 'S'])

        # all codons
        codons = [cod for cod in su.genetic_code.keys()]
        s = ''.join(codons)
        f1 = ''.join([su.genetic_code[cod] for cod in codons])

        s2 = s[1:-2]
        self.assertEqual(len(s2) % 3, 0)
        codons2 = [s2[i:i+3] for i in range(0, len(s2), 3)]
        f2 = ''.join([su.genetic_code[cod] for cod in codons2])

        s3 = s[2:-1]
        self.assertEqual(len(s3) % 3, 0)
        codons3 = [s3[i:i+3] for i in range(0, len(s3), 3)]
        f3 = ''.join([su.genetic_code[cod] for cod in codons3])

        s4 = Seq(s).reverse_complement()
        self.assertEqual(len(s4) % 3, 0)
        codons4 = [s4[i:i+3] for i in range(0, len(s4), 3)]
        f4 = ''.join([su.genetic_code[cod] for cod in codons4])

        s5 = s4[1:-2]
        self.assertEqual(len(s5) % 3, 0)
        codons5 = [s5[i:i+3] for i in range(0, len(s5), 3)]
        f5 = ''.join([su.genetic_code[cod] for cod in codons5])

        s6 = s4[2:-1]
        self.assertEqual(len(s6) % 3, 0)
        codons6 = [s6[i:i+3] for i in range(0, len(s6), 3)]
        f6 = ''.join([su.genetic_code[cod] for cod in codons6])

        self.assertEqual(su.six_frame_translation(s), [f1, f2, f3, f4, f5, f6])

    def test_convert_six_frame_position(self):
        for seqlen in [42, 43, 45]:
            f0aa, f0c = 0, 0  # keep track of aa position and codon position
            f1aa, f1c = -1, 2
            f2aa, f2c = -1, 1
            f3aa, f3c = 0, 0
            f4aa, f4c = -1, 2
            f5aa, f5c = -1, 1
            for fdnapos, rdnapos in enumerate(list(range(seqlen))[::-1]):
                if f0c > 2: # reset codon position and increment aa position
                    f0c = 0
                    f0aa += 1
                if f1c > 2:
                    f1c = 0
                    f1aa += 1
                if f2c > 2:
                    f2c = 0
                    f2aa += 1
                if f3c > 2:
                    f3c = 0
                    f3aa += 1
                if f4c > 2:
                    f4c = 0
                    f4aa += 1
                if f5c > 2:
                    f5c = 0
                    f5aa += 1

                self.assertEqual(su.convert_six_frame_position(fdnapos, 0, seqlen, dna_to_aa=True), f0aa, 
                                 f"dnapos={fdnapos}, seqlen={seqlen}, f0aa={f0aa}, f0c={f0c}")
                self.assertEqual(su.convert_six_frame_position(fdnapos, 1, seqlen, dna_to_aa=True), f1aa,
                                 f"dnapos={fdnapos}, seqlen={seqlen}, f1c={f1c}")
                self.assertEqual(su.convert_six_frame_position(fdnapos, 2, seqlen, dna_to_aa=True), f2aa,
                                 f"dnapos={fdnapos}, seqlen={seqlen}, f2c={f2c}")
                self.assertEqual(su.convert_six_frame_position(rdnapos, 3, seqlen, dna_to_aa=True), f3aa,
                                 f"dnapos={rdnapos}, seqlen={seqlen}, f3c={f3c}")
                self.assertEqual(su.convert_six_frame_position(rdnapos, 4, seqlen, dna_to_aa=True), f4aa,
                                 f"dnapos={rdnapos}, seqlen={seqlen}, f4c={f4c}")
                self.assertEqual(su.convert_six_frame_position(rdnapos, 5, seqlen, dna_to_aa=True), f5aa,
                                 f"dnapos={rdnapos}, seqlen={seqlen}, f5c={f5c}")
                
                f0c += 1 # increment codon position
                f1c += 1
                f2c += 1
                f3c += 1
                f4c += 1
                f5c += 1
            
            # test aa to dna
            f0dna = 0
            f1dna = 1
            f2dna = 2
            f3dna = seqlen - 1 - 2 # first codon base w.r.t. forward strand
            f4dna = seqlen - 2 - 2
            f5dna = seqlen - 3 - 2
            for aapos in range(seqlen):
                self.assertEqual(su.convert_six_frame_position(aapos, 0, seqlen, dna_to_aa=False), f0dna)
                self.assertEqual(su.convert_six_frame_position(aapos, 1, seqlen, dna_to_aa=False), f1dna)
                self.assertEqual(su.convert_six_frame_position(aapos, 2, seqlen, dna_to_aa=False), f2dna)
                self.assertEqual(su.convert_six_frame_position(aapos, 3, seqlen, dna_to_aa=False), f3dna)
                self.assertEqual(su.convert_six_frame_position(aapos, 4, seqlen, dna_to_aa=False), f4dna)
                self.assertEqual(su.convert_six_frame_position(aapos, 5, seqlen, dna_to_aa=False), f5dna)

                f0dna += 3
                f1dna += 3
                f2dna += 3
                f3dna -= 3
                f4dna -= 3
                f5dna -= 3
        

if __name__ == '__main__':
    unittest.main(verbosity=2)