#!/usr/bin/env python

import unittest
import numpy as np

from crane.neural_network import NeuralNetwork
from crane.expression import Expression

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.entrezs   = np.array([1,2,3,4])
        self.labels    = np.array([1,1,1,0,0,0])
        self.expression_data = np.array([[1,0,1,1,0,1],
                                         [0,1,0,1,0,1],
                                         [0,1,1,1,0,0],
                                         [0,0,0,0,1,1]])

        self.states = ((1,2,3), (2,4))
        self.expression = Expression(self.expression_data, self.entrezs, self.labels,
                binarize=False)

        self.n = NeuralNetwork(self.states)
        self.n.train(self.expression)

    def tearDown(self):
        pass

    def test_activate(self):
        sample_labels = (((1,0,0,0), 1),
                         ((0,1,1,0), 1),
                         ((0,0,0,1), 0),
                         ((1,1,0,1), 0))

        # for sample, label in sample_labels:
        #     output = self.n.activate(sample)
        #     self.assertEqual(output, label)

    def test_activate_with_entrez(self):
        sample_entrez_labels = (((1,0), (1,4), 1),
                                ((0,0), (1,4), 1),
                                ((0,1), (1,4), 0),
                                ((1,1), (1,4), 0))
        # for sample, entrezs, label in sample_entrez_labels:
        #     output = self.n.activate(sample, entrezs)
        #     self.assertEqual(output, label)

    def test_calc_feature(self):
        sample_entrez_expected = (((9,7),   (1,4),   (9,0,0,0,7)),
                                  ((9,7,8), (1,4,3), (9,0,8,0,7)))

        for sample, entrezs, expected in sample_entrez_expected:
             feature = self.n._calc_feature(sample, entrezs, missing=0)
             np.testing.assert_array_equal(feature, expected)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestNeuralNetwork))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=1).run(suite())
