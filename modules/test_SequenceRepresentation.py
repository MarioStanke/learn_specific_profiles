from Bio.Seq import Seq
import unittest

from . import SequenceRepresentation as sr

# TODO: write much more tests!

class TestSequence(unittest.TestCase):
    def setUp(self):
        # a number of dictionaries containing sequences and their attributes to test against sr.Sequence
        self.testdicts = [
            {
                "species": "Testspecies",
                "chr": "Testchr",
                "strand": "+",
                "start": 1100,
                "end": 1116,
                "len": 16,
                "sequence": "ATCGATCGATCGATCG",
                "type": "Testsequence",
                "exp_id": "Testspecies:Testchr:1,100-1,116"
            },
            {
                "species": "Testspecies",
                "chr": "Testchr",
                "strand": "-",
                "start": 1100,
                "end": 1116,
                "len": 16,
                "sequence": "ATCGATCGATCGATCG",
                "type": "Testsequence",
                "exp_id": "Testspecies:Testchr:1,100-1,116"
            }
        ]

    def tearDown(self):
        pass

    def test_Sequence(self):
        for testdict in self.testdicts:
            sequence = sr.Sequence(testdict["species"], testdict["chr"], testdict["strand"], testdict["start"], 
                                   testdict["end"], testdict["len"], testdict["sequence"], testdict["type"])
            self.assertEqual(sequence.species, testdict["species"])
            self.assertEqual(sequence.chromosome, testdict["chr"])
            self.assertEqual(sequence.strand, testdict["strand"])
            self.assertEqual(sequence.genome_start, testdict["start"])
            self.assertEqual(sequence.genome_end, testdict["end"])
            self.assertEqual(len(sequence), testdict["len"])
            self.assertEqual(sequence.getSequence(), testdict["sequence"])
            self.assertEqual(sequence.getSequence(rc=True), Seq(testdict["sequence"]).reverse_complement())
            self.assertEqual(sequence.type, testdict["type"])
            self.assertEqual(sequence.id, testdict["exp_id"])