#!/usr/bin/env python

import unittest
import pdb
import networkx as nx
import numpy as np

from crane.core import Core
from crane.expression import Expression

class TestCore(unittest.TestCase):
    def setUp(self):
        self.entrezs   = np.array([1,2])
        self.labels    = np.array([1,1,1,0,0,0])
        self.expr_data = np.array([[1,0,1,1,0,1],
                                   [0,1,0,1,0,1]])

        d = 2
        b = 10
        j = 0.2

        graph = nx.Graph()
        graph.add_edge(1,2)
        self.expression = Expression(self.expr_data, self.entrezs, self.labels, binarize=False)
        self.crane = Core(graph, self.expression, d, b, j)

    def test_run(self):
        self.crane.run()
        self.assertTrue(len(self.crane.T) == 2)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCore))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=1).run(suite())
