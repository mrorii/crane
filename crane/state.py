#!/usr/bin/env python

from expression import Expression
import numpy as np

class State:
    def __init__(self, genes, expr_ptn, expr_data):
        assert isinstance(genes, list)
        assert isinstance(expr_ptn, list)
        assert len(genes) == len(expr_ptn)

        self.genes     = genes
        self.expr_ptn  = expr_ptn
        self.expr_data = expr_data

        self.calced_info = False
        self.info        = None
        self.C           = (0, 1)

    def most_recent_gene(self):
        assert len(self.genes) > 0
        return self.genes[-1]

    def _calc_info(self, genes, expr_ptn, bound=False):
        assert len(genes) == len(expr_ptn)
        data = Expression(*self.expr_data.subset(genes), binarize=False)

        p_c = [float(data.num_samples(c)) / data.num_samples() for c in self.C]
        assert sum(p_c) == 1
        num_c_given_f = [0 for c in self.C]

        for c in self.C:
            num_f = 0

            equality = (expr_ptn == data.sample_gene).all(axis=1)
            for i, equal in enumerate(equality):
                if equal:
                    num_f += 1
                    if data.labels[i] == c:
                        num_c_given_f[c] += 1

        if num_f == 0: return 0
        num_f = float(num_f)

        s = 0
        if bound:
            s = max((num_c_given_f[c] / num_f) * np.log(1 / p_c[c])
                    if p_c[c] != 0 else 0 for c in self.C)
        else:
            s = sum((num_c_given_f[c] / num_f) * np.log((num_c_given_f[c] / num_f) / p_c[c])
                    if num_c_given_f[c] != 0 and p_c[c] != 0 else 0 for c in self.C)

        p_f = num_f / data.num_samples()
        retval = p_f * s
        assert retval >= 0.0
        return retval

    def _entropy(self, counts):
        ps = counts / float(np.sum(counts))
        ps = ps[np.nonzero(ps)]
        H = -np.sum(ps * np.log2(ps))
        return H

    def calc_mutual_info(self):
        '''Calculates (full) mutual information I(F_S;C)'''
        data = Expression(*self.expr_data.subset(self.genes), binarize=False)
        counts_C = np.histogram(data.labels, bins=[0,1,2])[0]

        num_genes = data.num_genes()
        bins = [[0,1,2] for i in xrange(num_genes)]
        counts_E, _ = np.histogramdd(data.sample_gene, bins=bins)
        counts_E = counts_E.ravel()

        # hack using binary to decimal conversion
        samples_as_decimals = [int(''.join(sample.astype('str')), base=2)
                for sample in data.sample_gene]
        counts_CE = np.histogram2d(data.labels, samples_as_decimals,
                bins=([0,1,2], range(2**num_genes + 1)))[0]

        H_CE = self._entropy(counts_CE)
        H_C  = self._entropy(counts_C)
        H_E  = self._entropy(counts_E)
        return H_C + H_E - H_CE

    def calc_info(self):
        '''Calculates (partial) mutual information J(f_S;C)'''
        if self.calced_info:
            return self.info

        self.calced_info = True
        self.info = self._calc_info(self.genes, self.expr_ptn)
        return self.info

    def calc_info_omit(self, gene):
        '''Calculates (partial) mutual information J(f_S\{g_j};C)'''
        assert gene in self.genes

        genes    = list(self.genes)    # clones list
        expr_ptn = list(self.expr_ptn) # clones list

        index = genes.index(gene)
        del genes[index]
        del expr_ptn[index]
        return self._calc_info(genes, expr_ptn)

    def calc_info_bound(self):
        '''Calculates (partial) mutual information J_{bound}(f_S;C)'''
        return self._calc_info(self.genes, self.expr_ptn, bound=True)

    def iter_gene(self):
        for gene in self.genes:
            yield gene

    def __contains__(self, gene):
        return gene in self.genes

    def __len__(self):
        return len(self.genes)

    def __str__(self):
        return '%s\t%s\t%.4f' % (self.genes, self.expr_ptn, self.calc_info())

    def __lt__(self, other):
        return self.calc_info() < other.calc_info()

