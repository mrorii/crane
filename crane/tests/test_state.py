#!/usr/bin/env python

import unittest
import numpy as np

from crane.expression import Expression
from crane.state import State

class TestState(unittest.TestCase):
    def setUp(self):
        all_entrezs   = np.array([1,2,3,4])
        self.labels    = np.array([1,1,1,0,0,0])
        self.expr_data = np.array([[1,0,1,1,0,1],
                                   [0,1,0,1,0,1],
                                   [0,1,1,1,0,0],
                                   [0,0,0,0,1,1]])

        self.expression = Expression(self.expr_data, all_entrezs, self.labels, binarize=False)
        self.entrezs  = [1,2]
        self.expr_ptn = [1,0]
        self.state = State(self.entrezs, self.expr_ptn, self.expression)

    def tearDown(self):
        pass

    def test_iter_gene(self):
        genes = list(self.state.iter_gene())
        np.testing.assert_array_equal(genes, self.entrezs)

    def test_calc_info(self):
        self.assertAlmostEqual(self.state.calc_info(), 1. / 3 * np.log(2))

    def test_calc_info_omit(self):
        self.assertEqual(self.state.calc_info_omit(2), 0)
        self.assertAlmostEqual(self.state.calc_info_omit(1),
                3. / 6 * (2. / 3 * np.log(2. / 3 * 2) + 1. / 3 * np.log(2. / 3)))
        self.assertGreater(self.state.calc_info(), self.state.calc_info_omit(1))
        self.assertGreater(self.state.calc_info(), self.state.calc_info_omit(2))

        self.assertEqual(self.state.calc_info_omit(self.entrezs[1]),
                State([self.entrezs[0]], [self.expr_ptn[0]], self.expression).calc_info())
        self.assertEqual(self.state.calc_info_omit(self.entrezs[0]),
                State([self.entrezs[1]], [self.expr_ptn[1]], self.expression).calc_info())

    def test_calc_info_bound(self):
        self.assertAlmostEqual(self.state.calc_info_bound(), 1. / 3 * np.log(2))

    def test_len(self):
        self.assertEqual(len(self.state), len(self.entrezs))

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestState))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=1).run(suite())

