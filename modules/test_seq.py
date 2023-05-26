import numpy as np
import random
import unittest

import seq
import sequtils as su

class TestSeq(unittest.TestCase):
    def test_getForbiddenPositions(self):
        self.assertEqual(seq.getForbiddenPositions([], k=3, slen=10), set([]))
        self.assertEqual(seq.getForbiddenPositions([0], k=3, slen=10), set([0,1,2]))
        self.assertEqual(seq.getForbiddenPositions([1], k=3, slen=10), set([0,1,2,3]))
        self.assertEqual(seq.getForbiddenPositions([2], k=3, slen=10), set([0,1,2,3,4]))
        self.assertEqual(seq.getForbiddenPositions([3], k=3, slen=10), set([1,2,3,4,5]))
        self.assertEqual(seq.getForbiddenPositions([4], k=3, slen=10), set([2,3,4,5,6]))
        self.assertEqual(seq.getForbiddenPositions([5], k=3, slen=10), set([3,4,5,6,7]))
        self.assertEqual(seq.getForbiddenPositions([6], k=3, slen=10), set([4,5,6,7,8]))
        self.assertEqual(seq.getForbiddenPositions([7], k=3, slen=10), set([5,6,7,8,9]))
        self.assertEqual(seq.getForbiddenPositions([8], k=3, slen=10), set([6,7,8,9]))
        self.assertEqual(seq.getForbiddenPositions([9], k=3, slen=10), set([7,8,9]))
        self.assertEqual(seq.getForbiddenPositions([0,1], k=3, slen=10), set([0,1,2,3]))
        self.assertEqual(seq.getForbiddenPositions([0,1,2], k=3, slen=10), set([0,1,2,3,4]))
        self.assertEqual(seq.getForbiddenPositions([0,1,2,3], k=3, slen=10), set([0,1,2,3,4,5]))
        self.assertEqual(seq.getForbiddenPositions([0,1,2,3,4], k=3, slen=10), set([0,1,2,3,4,5,6]))
        self.assertEqual(seq.getForbiddenPositions([0,1,2,3,4,5], k=3, slen=10), set([0,1,2,3,4,5,6,7]))
        self.assertEqual(seq.getForbiddenPositions([0,1,2,3,4,5,6], k=3, slen=10), set([0,1,2,3,4,5,6,7,8]))
        self.assertEqual(seq.getForbiddenPositions([0,1,2,3,4,5,6,7], k=3, slen=10), set([0,1,2,3,4,5,6,7,8,9]))
        self.assertEqual(seq.getForbiddenPositions([0,1,2,3,4,5,6,7,8], k=3, slen=10), set([0,1,2,3,4,5,6,7,8,9]))
        self.assertEqual(seq.getForbiddenPositions([0,1,2,3,4,5,6,7,8,9], k=3, slen=10), set([0,1,2,3,4,5,6,7,8,9]))
        self.assertEqual(seq.getForbiddenPositions([5], k=0, slen=10), set([]))
        self.assertEqual(seq.getForbiddenPositions([5], k=1, slen=10), set([5]))
        self.assertEqual(seq.getForbiddenPositions([5], k=2, slen=10), set([4,5,6]))
        self.assertEqual(seq.getForbiddenPositions([5], k=10, slen=10), set([0,1,2,3,4,5,6,7,8,9]))
        
    def test_insertPatternsToGenomes(self):
        self.skipTest("Not implemented")

    def test_getRandomGenomes(self):
        self.skipTest("Not implemented")

    def test_simulateGenomes(self):
        self.skipTest("Not implemented")

    def test_getNextBatch(self):
        self.skipTest("Not implemented")

    def test_backGroundAAFreqs(self):
        self.skipTest("Not implemented")
            

            
if __name__ == '__main__':
    unittest.main(verbosity=2)