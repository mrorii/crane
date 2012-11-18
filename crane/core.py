#!/usr/bin/env python

from state import State
import logging

class Core:
    def __init__(self, graph, expr_data, d, b, j):
        self.graph      = graph
        self.expr_data  = expr_data
        self.d          = d
        self.b          = b
        self.j          = j
        self.T          = []
        self.C          = (0, 1)

    def run(self):
        for i, node in enumerate(self.graph.nodes_iter()):
            if (i % 1000) == 0:
                logging.info('Node %5d completed (%s states)' % (i, len(self.T)))
            for e in self.C:
                state = State([node], [e], self.expr_data)
                self.extend_state(state)

    def extend_state(self, state):
        if len(state) == self.d:
            if state.calc_info() >= self.j:
                self.T.append(state)
            return

        Q = []
        g_i = state.most_recent_gene()

        for g_k in self.graph.neighbors_iter(g_i):
            if g_k < g_i: continue
            if g_k in state: continue

            for e in self.C:
                new_state = State(state.genes + [g_k], state.expr_ptn + [e], self.expr_data)
                redundant = False

                for g_j in state.iter_gene():
                    if new_state.calc_info_omit(g_j) >= new_state.calc_info():
                        redundant = True
                        break

                if not redundant and new_state.calc_info_bound() >= self.j:
                    Q.append(new_state)

        if not Q:
            if state.calc_info() >= self.j:
                self.T.append(state)
            return

        Q_b = sorted(Q, key=lambda s: - s.calc_info())[:self.b]

        for new_state in Q_b:
            self.extend_state(new_state)

