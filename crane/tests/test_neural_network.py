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

        for sample, label in sample_labels:
            output = self.n.activate(sample)
            self.assertEqual(output, label)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestNeuralNetwork))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=1).run(suite())
