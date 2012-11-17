#!/usr/bin/env python

import numpy as np
from sklearn.preprocessing import scale

class Expression:
    def __init__(self, expr_data, entrezs, labels, binarize=True, alpha=0.25):
        '''
        Creates a new expression object.
        `expr_data` is a 2-d numpy array with (entrez, sample) as dimensions
        '''
        expr_data = np.array(expr_data)
        entrezs   = np.array(entrezs)
        labels    = np.array(labels)
        assert len(entrezs) == expr_data.shape[0]
        assert len(labels)  == expr_data.shape[1]

        data = expr_data

        if binarize:
            # Binarize the expression dataset by first normalizing by gene
            # (average expression of 0 and std of 1)
            # Then set the top `alpha` fraction of the entries in the
            # normalized gene expression matrix to 1 and the rest to 0
            # In the paper, Chowdhury uses `alpha` = 0.25
            data = scale(data.astype(float), axis=1, with_mean=True, with_std=True)

            num_elements = data.size
            index = int(num_elements * (1 - alpha))
            threshold = data.flatten()[ data.argsort(axis=None)[index] ]

            lt_threshold = data < threshold
            gt_threshold = data >= threshold
            data[lt_threshold] = 0
            data[gt_threshold] = 1
            data = data.astype(int)

        self.gene_sample = data
        self.sample_gene = data.T

        self.entrezs = entrezs
        self.labels  = labels

        self.eid_to_index = {}
        for i, eid in enumerate(self.entrezs):
            self.eid_to_index[eid] = i

    def subset(self, eids):
        '''Returns the information necessary for creating a new Expression object
        represeting the subset (slice) of the original data for the specified entrez IDs'''
        assert isinstance(eids, list)
        indices = np.array([self.eid_to_index[eid] for eid in eids])

        return (self.gene_sample[indices], eids, self.labels)

    def num_genes(self):
        '''Returns the number of genes in this expression dataset'''
        return len(self.entrezs)

    def num_samples(self, c=None):
        '''Returns the number of patient samples in this expression dataset'''
        if isinstance(c, int):
            return np.sum(self.labels == c)
        else:
            return len(self.labels)

    def labels(self):
        '''Returns a numpy array representing the labels of samples in expression dataset'''
        return self.labels

