__author__ = 'pva701'

import tensorflow as tf

class OuterComposition:

    def __init__(self, outers, mode='PIPE'):
        self.outers = outers
        self.mode = mode

    def init_with_scope(self, window_vec_size, processed_vec_size):
        self.window_vec_size = window_vec_size
        self.init_proc_vec_size = processed_vec_size
        pr_size = processed_vec_size
        for outer in self.outers:
            outer.init_with_scope(window_vec_size, pr_size)
            if self.mode == 'PIPE':
                pr_size = outer.output_vector_size()
            elif self.mode == 'CONCAT':
                pr_size = processed_vec_size + outer.output_vector_size()

    def build_graph(self, *kargs):
        processed = kargs[0]
        out_rep = []
        out_rep.append(self.outers[0].build_graph(*kargs))
        for outer in self.outers[1:]:
            if self.mode == 'PIPE':
                out_rep.append(outer.build_graph(out_rep[-1], *kargs[1:]))
            elif self.mode == 'CONCAT':
                out_rep.append(outer.build_graph(tf.concat([processed, out_rep[-1]], 1), *kargs[1:]))
        return out_rep[-1]

    def output_vector_size(self):
        return self.outers[-1].output_vector_size()