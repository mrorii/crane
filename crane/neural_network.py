#!/usr/bin/env python

from itertools import izip
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
        self.feature_indices = np.array([index - 1 for sublist in states for index in sublist])

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

    def train(self, expression_data, labels):
        expression_data = np.array(expression_data)

        ds = SupervisedDataSet(self.num_features, 1)
        for sample, label in izip(expression_data.T, labels):
            feature = self._calc_feature(sample)
            ds.addSample(feature, (label,))

        # train
        self.trainer = BackpropTrainer(self.n, ds)
        self.trainer.trainUntilConvergence()

        self.expression_data = expression_data
        self.ds              = ds

    def activate(self, sample):
        assert len(sample) == self.expression_data.shape[0]
        sample  = np.array(sample)
        feature = self._calc_feature(sample)
        return int(np.round(self.n.activate(feature)))

    def _calc_feature(self, sample):
        return sample[ self.feature_indices ]

