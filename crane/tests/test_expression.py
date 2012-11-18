#!/usr/bin/env python

import unittest
import numpy as np

from crane.expression import Expression

class TestExpression(unittest.TestCase):
    def setUp(self):
        self.entrezs   = np.array([1,2])
        self.labels    = np.array([1,1,1,0,0,0])
        self.expr_data = np.array([[1,0,1,1,0,1],
                                   [0,1,0,1,0,1]])

        self.expression = Expression(self.expr_data, self.entrezs, self.labels, binarize=False)

    def tearDown(self):
        pass

    def test_init(self):
        np.testing.assert_array_equal(self.expression.gene_sample, self.expr_data)
        np.testing.assert_array_equal(self.expression.sample_gene, self.expr_data.T)

    def test_num_genes(self):
        self.assertEqual(self.expression.num_genes(), len(self.entrezs))

    def test_num_samples(self):
        self.assertEqual(self.expression.num_samples(), len(self.labels))
        self.assertEqual(self.expression.num_samples(1), np.sum(self.labels == 1))
        self.assertEqual(self.expression.num_samples(0), np.sum(self.labels == 0))

    def test_subset(self):
        eids = [1]
        g, e, l = self.expression.subset(eids)
        np.testing.assert_array_equal(g, np.array([self.expr_data[0]]))
        np.testing.assert_array_equal(e, eids)
        np.testing.assert_array_equal(l, self.labels)

    def test_subset_clone(self):
        expression = Expression(*self.expression.subset(list(self.entrezs)), binarize=False)
        self.assertEqual(expression.num_samples(), len(self.labels))
        self.assertEqual(expression.num_genes(), len(self.entrezs))
        np.testing.assert_array_equal(expression.gene_sample, self.expr_data)

    def test_binarize(self):
        alpha = 0.25
        expression = Expression(self.expr_data, self.entrezs, self.labels, binarize=True,
                alpha=alpha)
        self.assertEqual(expression.gene_sample.sum(), int(expression.gene_sample.size * alpha))

    def test_index_of_gene(self):
        self.assertEqual(self.expression.index_of_gene(2), 1)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestExpression))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=1).run(suite())

