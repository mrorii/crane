#!/usr/bin/env python

import numpy as np

from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules import BiasUnit, LinearLayer, TanhLayer
from pybrain.structure.connections import FullConnection, IdentityConnection

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet

class NeuralNetwork:
    def __init__(self, states):
        '''Create a NeuralNetwork instance.

        `states` is a tuple of tuples of ints, representing the discovered subnetworks'
        entrez ids.
        '''
        self.num_features    = sum(map(lambda tup: len(tup), states))
        self.states          = states

        n = FeedForwardNetwork()
        n.addOutputModule(TanhLayer(1, name='out'))
        n.addModule(BiasUnit(name='bias out'))
        n.addConnection(FullConnection(n['bias out'], n['out']))

        for i, state in enumerate(states):
            dim = len(state)
            n.addInputModule(TanhLayer(dim, name='input %s' % i))
            n.addModule(BiasUnit(name='bias input %s' % i))
            n.addConnection(FullConnection(n['bias input %s' % i], n['input %s' % i]))
            n.addConnection(FullConnection(n['input %s' % i], n['out']))

        n.sortModules()
        self.n = n

    def train(self, expression):
        self.expression = expression
        self.feature_indices = self._calc_feature_indices()
        ds = SupervisedDataSet(self.num_features, 1)
        for sample, label in expression.iter_sample_label():
            feature = self._calc_feature(sample)
            ds.addSample(feature, (label,))

        self.trainer = BackpropTrainer(self.n, ds)
        self.trainer.trainUntilConvergence()

        self.ds = ds

    def activate(self, sample):
        assert len(sample) == self.expression.num_genes()
        sample  = np.array(sample)
        feature = self._calc_feature(sample)
        return int(np.round(self.n.activate(feature)))

    def _calc_feature_indices(self):
        feature_indices = []
        for substate in self.states:
            for eid in substate:
                index = self.expression.index_of_gene(eid)
                feature_indices.append(index)
        return np.array(feature_indices)

    def _calc_feature(self, sample):
        return sample[self.feature_indices]

