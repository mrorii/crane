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
        self._calc_feature_indices()

        ds = SupervisedDataSet(self.num_features, 1)
        for sample, label in expression.iter_sample_label():
            feature = self._calc_feature(sample)
            ds.addSample(feature, (label,))

        self.trainer = BackpropTrainer(self.n, dataset=ds, learningrate=0.01, lrdecay=1.0,
                momentum=0.1, verbose=True)
        self.trainer.trainUntilConvergence()

        self.ds = ds

    def activate(self, sample, entrezs=[], missing=0):
        '''Fires the neural network with the given sample.
        By default, assumes that the entrez IDs for the `sample` matches the
        expression data used for training.

        If the entrez IDs differ from the training expression data,
        it can be specified by the list `entrezs`.

        `missing` specifies the expression value for missing entrez IDs.
        '''
        sample  = np.array(sample)
        feature = self._calc_feature(sample, entrezs=entrezs, missing=missing)
        return self.n.activate(feature)

    def _calc_feature_indices(self):
        feature_indices = []
        feature_eids    = []
        for substate in self.states:
            for eid in substate:
                index = self.expression.index_of_gene(eid)
                feature_indices.append(index)
                feature_eids.append(eid)
        self.feature_indices = np.array(feature_indices)
        self.feature_eids    = np.array(feature_eids)

    def _calc_feature(self, sample, entrezs=[], missing=0):
        if not entrezs:
            # Use the indices from the training expression data
            return sample[self.feature_indices]
        else:
            feature = []

            for feature_eid in self.feature_eids:
                if feature_eid in entrezs:
                    index = entrezs.index(feature_eid)
                    feature.append( sample[index] )
                else:
                    feature.append(missing)
            return np.array(feature)

