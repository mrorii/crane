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

    def most_recent_gene(self):
        assert len(self.genes) > 0
        return self.genes[-1]

    def _calc_info(self, genes, expr_ptn, bound=False):
        assert len(genes) == len(expr_ptn)

        data = Expression(*self.expr_data.subset(genes), binarize=False)

        s = 0
        C = (0, 1)

        p_c = [float(data.num_samples(c)) / data.num_samples() for c in C]
        assert sum(p_c) == 1
        num_c_given_f = [0 for c in C]
        for c in C:
            num_f = 0

            equality = (expr_ptn == data.sample_gene).all(axis=1)
            for i, equal in enumerate(equality):
                if equal:
                    num_f += 1
                    if data.labels[i] == c:
                        num_c_given_f[c] += 1

        if num_f == 0: return 0
        num_f = float(num_f)

        if bound:
            s = max((num_c_given_f[c] / num_f) * np.log(1 / p_c[c])
                    if p_c[c] != 0 else 0 for c in C)
        else:
            s = sum((num_c_given_f[c] / num_f) * np.log((num_c_given_f[c] / num_f) / p_c[c])
                    if num_c_given_f[c] != 0 and p_c[c] != 0 else 0 for c in C)

        p_f = num_f / data.num_samples()
        retval = p_f * s
        assert retval >= 0.0
        return retval

    def calc_info(self):
        if self.calced_info:
            return self.info

        self.calced_info = True
        self.info = self._calc_info(self.genes, self.expr_ptn)
        return self.info

    def calc_info_omit(self, gene):
        assert gene in self.genes

        genes    = list(self.genes) # clones list
        expr_ptn = list(self.expr_ptn) # clones list

        index = genes.index(gene)
        del genes[index]
        del expr_ptn[index]
        return self._calc_info(genes, expr_ptn)

    def calc_info_bound(self):
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

