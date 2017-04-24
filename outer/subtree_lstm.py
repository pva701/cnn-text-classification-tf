__author__ = 'pva701'

import tensorflow as tf
from tensorflow.contrib import rnn
import utils as tfu

class SubtreeLstm:

    def init_with_scope(self, window_vec_size, processed_vec_size):
        self.hidden_size = processed_vec_size
        self.in_size = window_vec_size
        with tf.variable_scope("subtree-lstm") as scope:
            self.cell, self.initial_state = \
                tfu.create_lstm(self.hidden_size, self.in_size + self.hidden_size, scope)

    def build_graph(self, out_repr, windows_repr, n_words,
                    _1, _2, l_bound, r_bound, _3, _4, _5, dropout_keep):
        def apply_to_inner(i):
            vector = out_repr[i + n_words]
            l_b = l_bound[i]
            r_b = r_bound[i]
            return self.fn(vector, windows_repr[l_b:r_b], r_b - l_b, dropout_keep)

        inner_vectors = tf.map_fn(apply_to_inner, tf.range(0, n_words - 1), dtype=tf.float32)
        return tf.concat([out_repr[:n_words], inner_vectors], 0)

    def fn(self, init_state, word_vecs, sub_n_words, dropout_keep_prob):
        with tf.variable_scope("subtree-lstm") as scope:
            scope.reuse_variables()

            def apply_children(state, i):
                _, new_state = self.cell(tf.expand_dims(tf.concat([init_state, word_vecs[i]], axis=0), 0),
                                         rnn.LSTMStateTuple(state[0], state[1]),
                                         scope)
                c, h = new_state
                return tf.stack([c, h])

            ret = tf.foldl(apply_children,
                           tf.range(tf.constant(0), sub_n_words),
                           initializer=(tf.zeros([1, self.hidden_size]), tf.zeros([1, self.hidden_size])))  # memory
            # initializer=(tf.expand_dims(init_state, 0), tf.zeros([1, self.hidden_size]))) #memory
            # initializer=(tf.zeros([1, self.hidden_size]), tf.expand_dims(init_state, 0))) #hidden
            return tf.reshape(ret[1], [-1])
            # Add dropout
            # with tf.name_scope("dropout"):
            #     return tf.reshape(tf.nn.dropout(ret[1], dropout_keep_prob), [-1])

    def output_vector_size(self):
        return self.hidden_size

    def l2_loss(self):
        ret = tf.constant(0.0)
        with tf.variable_scope("subtree-lstm") as scope:
            scope.reuse_variables()
            for var in tf.trainable_variables():
                if "subtree-lstm/" in var.name:
                    ret += tf.nn.l2_loss(var)
        return ret
