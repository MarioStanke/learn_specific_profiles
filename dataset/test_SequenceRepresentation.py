import json
import os
import unittest

import SequenceRepresentation as sr

class TestSequenceRepresentation(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        self.sequence = sr.Sequence(species='testspecies', chromosome='testchromosome', strand='-', genome_start=42, 
                                    genome_end=66, length=24, sequence='ATCGATCGATCGATCGATCGATCG', 
                                    seqtype='testseqtype', no_homology=False, no_elements=False)
        # infer length from end position
        self.sequence_end = sr.Sequence(species='testspecies', chromosome='testchromosome', strand='-', genome_start=42,
                                        genome_end=66, seqtype='testseqtype', no_homology=False, no_elements=False)
        # infer end position from length
        self.sequence_len = sr.Sequence(species='testspecies', chromosome='testchromosome', strand='-', genome_start=42,
                                        length=24, seqtype='testseqtype', no_homology=False, no_elements=False)
        # infer length and end from sequence
        self.sequence_seq = sr.Sequence(species='testspecies', chromosome='testchromosome', strand='-', genome_start=42,
                                        sequence='ATCGATCGATCGATCGATCGATCG', seqtype='testseqtype', no_homology=False,
                                        no_elements=False)
        
        # setup elements
        self.elem_other_species = sr.Sequence('otherspecies', 'testchromosome', '+', 42, 66, no_homology=True)
        self.elem_before = sr.Sequence('testspecies', 'testchromosome', '+', 0, 42)
        self.elem_overlap_left = sr.Sequence('testspecies', 'testchromosome', '-', 0, 43, seqtype='overlap_left')
        self.elem_inside = sr.Sequence('testspecies', 'testchromosome', '+', 43, 65, seqtype='inside')
        self.elem_overlap_both = sr.Sequence('testspecies', 'testchromosome', '-', 41, 67, seqtype='overlap_both', 
                                             no_elements=True)
        self.elem_overlap_right = sr.Sequence('testspecies', 'testchromosome', '-', 65, 420, seqtype='overlap_right')
        self.elem_after = sr.Sequence('testspecies', 'testchromosome', '+', 66, 420)

        # setup dict representation
        self.seqdict = {
            "id": "testspecies:testchromosome:42-66",
            "species": "testspecies",	
            "chromosome": "testchromosome",
            "strand": "-",
            "genome_start": 42,
            "genome_end": 66,
            "length": 24,
            "sequence": "ATCGATCGATCGATCGATCGATCG",
            "type": "testseqtype",
            "genomic_elements": [
                {
                    "id": "testspecies:testchromosome:41-67",
                    "species": "testspecies",
                    "chromosome": "testchromosome",
                    "strand": "-",
                    "genome_start": 41,
                    "genome_end": 67,
                    "length": 26,
                    "type": "overlap_both",
                    "homology": []
                }
            ],
            "homology": [
                {
                    "id": "otherspecies:testchromosome:42-66",
                    "species": "otherspecies",
                    "chromosome": "testchromosome",
                    "strand": "+",
                    "genome_start": 42,
                    "genome_end": 66,
                    "length": 24,
                    "type": "sequence",
                    "genomic_elements": []
                }
            ]
        }

        # try to create a temporary test file
        self.testfile = 'testfile.json'
        if not os.path.exists(self.testfile):
            open(self.testfile, 'a').close() # create empty file
            self.testfile_created = True
        else:
            self.testfile_created = False

    def tearDown(self) -> None:
        # remove temporary test file
        if self.testfile_created:
            os.remove(self.testfile)

    def test_attributes(self):
        for sequence in [self.sequence, self.sequence_end, self.sequence_len, self.sequence_seq]:
            with self.subTest("sequence hasattr"):
                self.assertTrue(hasattr(sequence, 'id'))
                self.assertTrue(hasattr(sequence, 'species'))
                self.assertTrue(hasattr(sequence, 'chromosome'))
                self.assertTrue(hasattr(sequence, 'strand'))
                self.assertTrue(hasattr(sequence, 'genome_start'))
                self.assertTrue(hasattr(sequence, 'genome_end'))
                self.assertTrue(hasattr(sequence, 'length'))
                self.assertTrue(hasattr(sequence, 'sequence'))
                self.assertTrue(hasattr(sequence, 'type'))
                self.assertTrue(hasattr(sequence, 'homology'))
                self.assertTrue(hasattr(sequence, 'genomic_elements'))

                self.assertEqual(sequence.id, 'testspecies:testchromosome:42-66')
                self.assertEqual(sequence.species, 'testspecies')
                self.assertEqual(sequence.chromosome, 'testchromosome')
                self.assertEqual(sequence.strand, '-')
                self.assertEqual(sequence.genome_start, 42)
                self.assertEqual(sequence.genome_end, 66)
                self.assertEqual(sequence.length, 24)
                self.assertEqual(len(sequence), 24)
                if sequence is self.sequence or sequence is self.sequence_seq:
                    self.assertEqual(len(sequence.sequence), 24)
                    self.assertEqual(sequence.sequence, 'ATCGATCGATCGATCGATCGATCG')
                else:
                    self.assertIsNone(sequence.sequence)

                self.assertEqual(sequence.type, 'testseqtype')
                self.assertEqual(sequence.homology, [])
                self.assertEqual(sequence.genomic_elements, [])

    def test_equal(self):
        sequence2 = sr.Sequence(species='testspecies', chromosome='testchromosome', strand='-', genome_start=42, 
                                genome_end=66, length=24, sequence='ATCGATCGATCGATCGATCGATCG', 
                                seqtype='testseqtype', no_homology=False, no_elements=False) # same as self.sequence
        self.assertEqual(self.sequence, sequence2)
        sequence2.addElement(self.elem_overlap_left) # alter sequence2
        self.assertNotEqual(self.sequence, sequence2)

    def test_addElementAndHomology(self):
        self.assertRaises(AssertionError, self.sequence.addElement, 'testelement')
        
        self.assertRaises(AssertionError, self.sequence.addElement, self.elem_before)
        self.assertRaises(AssertionError, self.sequence.addElement, self.elem_after)
        self.assertRaises(AssertionError, self.sequence.addElement, self.elem_other_species)
        self.assertFalse(self.sequence.hasElements())
        self.assertTrue(self.sequence.elementsPossible())
        self.sequence.addElement(self.elem_overlap_left)
        self.sequence.addElement(self.elem_inside)
        self.sequence.addElement(self.elem_overlap_both)
        self.sequence.addElement(self.elem_overlap_right)
        self.assertEqual(len(self.sequence.genomic_elements), 4)
        self.assertTrue(self.sequence.hasElements())
        self.assertEqual(self.sequence.genomic_elements[0], self.elem_overlap_left)
        self.assertEqual(self.sequence.genomic_elements[1], self.elem_inside)
        self.assertEqual(self.sequence.genomic_elements[2], self.elem_overlap_both)
        self.assertEqual(self.sequence.genomic_elements[3], self.elem_overlap_right)
        self.assertRaises(AssertionError, self.elem_overlap_both.addElement, self.sequence)
        self.assertFalse(self.elem_overlap_both.hasElements())
        self.assertFalse(self.elem_overlap_both.elementsPossible())

        self.assertFalse(self.sequence.hasHomologies())
        self.assertTrue(self.sequence.homologiesPossible())
        self.sequence.addHomology(self.elem_other_species)
        self.assertEqual(len(self.sequence.homology), 1)
        self.assertTrue(self.sequence.hasHomologies())
        self.assertRaises(AssertionError, self.elem_other_species.addHomology, self.sequence)
        self.assertFalse(self.elem_other_species.hasHomologies())
        self.assertFalse(self.elem_other_species.homologiesPossible())

    def test_addSubsequenceAsElement(self):
        self.assertFalse(self.sequence.hasElements())
        self.assertTrue(self.sequence.elementsPossible())
        self.assertRaises(AssertionError, self.sequence.addSubsequenceAsElement, 0, 42, "before", 
                          genomic_positions=True)
        self.assertRaises(AssertionError, self.sequence.addSubsequenceAsElement, 66, 420, "after", 
                          genomic_positions=True)
        self.sequence.addSubsequenceAsElement(0, 43, "overlap_left", genomic_positions=True)
        self.sequence.addSubsequenceAsElement(43, 65, "inside", "+", genomic_positions=True)
        self.sequence.addSubsequenceAsElement(41, 67, "overlap_both", genomic_positions=True, no_elements=True)
        self.sequence.addSubsequenceAsElement(65, 420, "overlap_right", genomic_positions=True)
        self.assertEqual(len(self.sequence.genomic_elements), 4)
        # add sequences to elements
        self.elem_overlap_left.sequence = "A"
        self.elem_inside.sequence = "TCGATCGATCGATCGATCGATC"
        self.elem_overlap_both.sequence = "ATCGATCGATCGATCGATCGATCG"
        self.elem_overlap_right.sequence = "G"
        self.assertEqual(self.sequence.genomic_elements[0], self.elem_overlap_left)
        self.assertEqual(self.sequence.genomic_elements[1], self.elem_inside)
        self.assertEqual(self.sequence.genomic_elements[2], self.elem_overlap_both)
        self.assertEqual(self.sequence.genomic_elements[3], self.elem_overlap_right)

    def test_addSubsequenceAsElementRelative(self):
        self.assertFalse(self.sequence.hasElements())
        self.assertTrue(self.sequence.elementsPossible())
        self.assertRaises(AssertionError, self.sequence.addSubsequenceAsElement, -42, 0, "before", 
                          genomic_positions=False)
        self.assertRaises(AssertionError, self.sequence.addSubsequenceAsElement, 24, 378, "after", 
                          genomic_positions=False)
        self.sequence.addSubsequenceAsElement(-42, 1, "overlap_left", genomic_positions=False)
        self.sequence.addSubsequenceAsElement(1, 23, "inside", "+", genomic_positions=False)
        self.sequence.addSubsequenceAsElement(-1, 25, "overlap_both", genomic_positions=False, no_elements=True)
        self.sequence.addSubsequenceAsElement(23, 378, "overlap_right", genomic_positions=False)
        self.assertEqual(len(self.sequence.genomic_elements), 4)
        # add sequences to elements
        self.elem_overlap_left.sequence = "A"
        self.elem_inside.sequence = "TCGATCGATCGATCGATCGATC"
        self.elem_overlap_both.sequence = "ATCGATCGATCGATCGATCGATCG"
        self.elem_overlap_right.sequence = "G"
        self.assertEqual(self.sequence.genomic_elements[0], self.elem_overlap_left)
        self.assertEqual(self.sequence.genomic_elements[1], self.elem_inside)
        self.assertEqual(self.sequence.genomic_elements[2], self.elem_overlap_both)
        self.assertEqual(self.sequence.genomic_elements[3], self.elem_overlap_right)

    def test_getRelativePositions(self):
        self.assertRaises(AssertionError, self.sequence.getRelativePositions, self.elem_before)
        self.assertRaises(AssertionError, self.sequence.getRelativePositions, self.elem_before, True)
        self.assertEqual(self.elem_overlap_left.getRelativePositions(self.sequence), (-42, 1))
        self.assertEqual(self.elem_overlap_left.getRelativePositions(self.sequence, True), (23, 66))
        self.assertEqual(self.elem_overlap_both.getRelativePositions(self.sequence), (-1, 25))
        self.assertEqual(self.elem_overlap_both.getRelativePositions(self.sequence, True), (-1, 25))
        self.assertEqual(self.elem_inside.getRelativePositions(self.sequence), (1, 23))
        self.assertEqual(self.elem_inside.getRelativePositions(self.sequence, True), (1, 23))
        self.assertEqual(self.elem_overlap_right.getRelativePositions(self.sequence), (23, 378))
        self.assertEqual(self.elem_overlap_right.getRelativePositions(self.sequence, True), (-354, 1))
        inside_left = sr.Sequence("testspecies", "testchromosome", "+", 46, 54) # inside left of sequence
        self.sequence.addElement(inside_left)
        self.assertEqual(inside_left.getRelativePositions(self.sequence), (4, 12))
        self.assertEqual(inside_left.getRelativePositions(self.sequence, True), (12, 20))
        # trick to get RC coordinates on "chromosome": create "chromosome" Sequence() as parent
        chromosome = sr.Sequence("testspecies", "testchromosome", "+", 0, 420)
        self.assertEqual(self.sequence.getRelativePositions(chromosome), (42, 66))
        self.assertEqual(self.sequence.getRelativePositions(chromosome, True), (354, 378))

    def test_getSequence(self):
        self.assertEqual(self.sequence.getSequence(), "ATCGATCGATCGATCGATCGATCG")
        self.assertEqual(self.sequence.getSequence(True), "CGATCGATCGATCGATCGATCGAT")
        self.assertIsNone(self.elem_other_species.getSequence())
        self.assertIsNone(self.elem_other_species.getSequence(True))
        self.sequence.addElement(self.elem_overlap_both)
        self.sequence.addSubsequenceAsElement(41, 67, "overlap_both", genomic_positions=True, no_elements=True)
        self.assertIsNone(self.sequence.genomic_elements[0].getSequence())
        self.assertIsNone(self.sequence.genomic_elements[0].getSequence(True))
        self.assertEqual(self.sequence.genomic_elements[1].getSequence(), "ATCGATCGATCGATCGATCGATCG")
        self.assertEqual(self.sequence.genomic_elements[1].getSequence(True), "CGATCGATCGATCGATCGATCGAT")

    def test_getSubsequence(self):
        self.assertIsNone(self.elem_other_species.getSubsequence(42, 66))
        self.assertIsNone(self.elem_other_species.getSubsequence(42, 66, True))
        self.assertEqual(self.sequence.getSubsequence(42, 66), "ATCGATCGATCGATCGATCGATCG")
        self.assertEqual(self.sequence.getSubsequence(42, 66, True), "CGATCGATCGATCGATCGATCGAT")
        self.assertEqual(self.sequence.getSubsequence(0, 43), "A")
        self.assertEqual(self.sequence.getSubsequence(0, 46, True), "CGAT")
        self.assertEqual(self.sequence.getSubsequence(65, 420), "G")
        self.assertEqual(self.sequence.getSubsequence(62, 420, True), "CGAT")
        # self.sequence not well suited to test rc behaviour
        sequence2 = sr.Sequence("testspecies", "testchromosome", "+", 0, sequence="AAAACCCCGTGTGTGT")
        self.assertEqual(sequence2.getSubsequence(2, 6), "AACC")
        self.assertEqual(sequence2.getSubsequence(2, 6, True), "GGTT")
        self.assertEqual(sequence2.getSubsequence(8, 12), "GTGT")
        self.assertEqual(sequence2.getSubsequence(8, 12, True), "ACAC")

    def test_sequencesOverlap(self):
        self.assertTrue(sr._sequencesOverlap(self.sequence, self.elem_overlap_left))
        self.assertTrue(sr._sequencesOverlap(self.sequence, self.elem_inside))
        self.assertTrue(sr._sequencesOverlap(self.sequence, self.elem_overlap_both))
        self.assertTrue(sr._sequencesOverlap(self.sequence, self.elem_overlap_right))
        self.assertFalse(sr._sequencesOverlap(self.sequence, self.elem_before))
        self.assertFalse(sr._sequencesOverlap(self.sequence, self.elem_after))
        self.assertFalse(sr._sequencesOverlap(self.sequence, self.elem_other_species))

        self.assertTrue(sr._sequencesOverlap(self.elem_overlap_left, self.sequence))
        self.assertTrue(sr._sequencesOverlap(self.elem_inside, self.sequence))
        self.assertTrue(sr._sequencesOverlap(self.elem_overlap_both, self.sequence))
        self.assertTrue(sr._sequencesOverlap(self.elem_overlap_right, self.sequence))
        self.assertFalse(sr._sequencesOverlap(self.elem_before, self.sequence))
        self.assertFalse(sr._sequencesOverlap(self.elem_after, self.sequence))
        self.assertFalse(sr._sequencesOverlap(self.elem_other_species, self.sequence))

    def test_stripSequences(self):
        self.sequence.addElement(self.elem_overlap_left)
        self.sequence.addElement(self.elem_overlap_both)
        self.assertEqual(len(self.sequence.genomic_elements), 2)
        self.sequence.stripSequence(3)
        self.assertEqual(self.sequence.getSequence(), "GATCGATCGATCGATCGATCG")
        self.assertEqual(self.sequence.length, 21)
        self.assertEqual(self.sequence.genome_start, 45)
        self.assertEqual(self.sequence.genome_end, 66)
        self.assertEqual(len(self.sequence.genomic_elements), 1)
        self.assertEqual(self.sequence.genomic_elements[0], self.elem_overlap_both)

        self.sequence.addElement(self.elem_overlap_right)
        self.assertEqual(len(self.sequence.genomic_elements), 2)
        self.sequence.stripSequence(3, from_start=False)
        self.assertEqual(self.sequence.getSequence(), "GATCGATCGATCGATCGA")
        self.assertEqual(self.sequence.length, 18)
        self.assertEqual(self.sequence.genome_start, 45)
        self.assertEqual(self.sequence.genome_end, 63)
        self.assertEqual(len(self.sequence.genomic_elements), 1)
        self.assertEqual(self.sequence.genomic_elements[0], self.elem_overlap_both)

    def test_toDict(self):
        self.sequence.addElement(self.elem_overlap_both)
        self.sequence.addHomology(self.elem_other_species)
        self.assertEqual(self.sequence.toDict(), self.seqdict)

    def test_fromJSON(self):
        seq = sr.sequenceFromJSON(jsonstring = json.dumps(self.seqdict))
        self.sequence.addElement(self.elem_overlap_both)
        self.sequence.addHomology(self.elem_other_species)
        self.assertEqual(seq, self.sequence)

    def test_fromJSONfile(self):
        if self.testfile_created:
            self.sequence.addElement(self.elem_overlap_both)
            self.sequence.addHomology(self.elem_other_species)
            with open(self.testfile, "w") as f:
                json.dump(self.sequence.toDict(), f)

            seq = sr.sequenceFromJSON(jsonfile = self.testfile)
            self.assertEqual(seq, self.sequence)
        else:
            self.skipTest("Testfile could not be created")

    def test_loadJSONlist(self):
        if self.testfile_created:
            seqlist = [self.sequence, self.elem_other_species]
            dictlist = [s.toDict() for s in seqlist]
            with open(self.testfile, "w") as f:
                json.dump(dictlist, f)

            loadedlist = sr.loadJSONSequenceList(self.testfile)
            self.assertEqual(loadedlist, seqlist)
        else:
            self.skipTest("Testfile could not be created")


if __name__ == '__main__':
    unittest.main(verbosity=2)