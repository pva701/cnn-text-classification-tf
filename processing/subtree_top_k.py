__author__ = 'pva701'

import tensorflow as tf
import utils as tfu
from tensorflow.contrib import rnn

FUN_ACT = tf.sigmoid

class SubtreeTopK:
    def __init__(self, k, mode='processing',
                 leaf_size=None,
                 backend=None,
                 num_filters=None,
                 filter_sizes=None,
                 lstm_hidden=None):
        self.k = k
        self.mode = mode
        self.leaf_size = leaf_size
        self.backend = backend
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.lstm_hidden = lstm_hidden
        if self.backend != 'SUM' and self.backend != 'CNN' and self.backend != 'LSTM':
            raise Exception('Unexpected SubtreeInnerTopK backend')

        if mode == 'processing':
            self.build_graph = self.__processing_build_graph
        elif mode == 'outer':
            self.build_graph = self.__outer_build_graph
        else:
            raise Exception('Mda')

        self.var_scope = "subtree-top-k-{}".format(self.mode)

    def init_with_scope(self, in_size):
        self.in_size = in_size
        with tf.variable_scope(self.var_scope) as scope:
            if self.backend == 'SUM':
                tfu.linear(self.k, 1, "sum")
            elif self.backend == 'CNN':
                if self.filter_sizes is None:
                    self.filter_sizes = [self.k]
                self.leaf_size = self.output_vector_size() if self.mode == 'processing' else None
                tfu.create_cnn(self.filter_sizes, self.num_filters, self.leaf_size or self.in_size)
            elif self.backend == 'LSTM':
                self.leaf_size = self.lstm_hidden if self.mode == 'processing' else None
                self.cell, self.initial_state = tfu.create_lstm(self.lstm_hidden, self.leaf_size or self.in_size, scope)

            if self.mode == 'processing':
                self.real_in_size = self.leaf_size or self.in_size
                if self.leaf_size:
                    tfu.linear(in_size, self.leaf_size, "top")
            else:
                self.real_in_size = self.in_size

    def __outer_build_graph(self, n_words, _0, out_repr, _1, _2, _3, _4, euler, euler_l, euler_r, dropout_keep):
        leaves_vectors = tf.map_fn(lambda v: self.subtree_fn(tf.expand_dims(v, 0)), out_repr[:n_words],
                                   dtype=tf.float32)

        inner_vectors = tf.map_fn(lambda i: self.subtree_fn(tf.gather(out_repr, euler[euler_l[i]:euler_r[i] - 1])),
                                  tf.range(0, n_words - 1), dtype=tf.float32)

        return tf.concat([leaves_vectors, inner_vectors], 0)

    def __processing_build_graph(self, n_words, words_vecs, _1, _2, _3, _4, euler, euler_l, euler_r, dropout_keep):
        with tf.variable_scope(self.var_scope) as scope:
            scope.reuse_variables()

            if self.leaf_size:
                W_top = tf.get_variable("W_top")
                biases_top = tf.get_variable("biases_top")
                # nn.tanh()
                leaves_vectors = FUN_ACT(tf.nn.xw_plus_b(words_vecs, W_top, biases_top))
            else:
                leaves_vectors = words_vecs

            def apply_top_k(vectors, i):
                nodes_vec = tf.gather(vectors, euler[euler_l[i]:euler_r[i] - 1])
                top_k_vec = tf.expand_dims(self.subtree_fn(nodes_vec), 0)
                return tf.concat([vectors, top_k_vec], 0)

        return tf.foldl(apply_top_k, tf.range(0, n_words - 1), initializer=leaves_vectors)

    def subtree_fn(self, subtree_vecs):
        with tf.variable_scope(self.var_scope) as scope:
            scope.reuse_variables()
            extended_subtree_vecs = tf.concat([subtree_vecs, tf.zeros([self.k, self.real_in_size])], 0)
            vec_len = tf.reduce_sum(tf.square(extended_subtree_vecs), 1)
            _, indices = tf.nn.top_k(vec_len, self.k, False, "top-vectors")
            res_vecs = tf.gather(extended_subtree_vecs, indices)

            if self.backend == 'SUM':
                W_sum = tf.get_variable("W_sum")
                biases_sum = tf.get_variable("biases_sum")
                return tf.transpose(FUN_ACT(tf.nn.xw_plus_b(tf.transpose(res_vecs), W_sum, biases_sum)))
            elif self.backend == 'CNN':
                return tfu.pass_cnn(self.filter_sizes,
                                    self.num_filters,
                                    res_vecs,
                                    self.real_in_size,
                                    seq_len=self.k,
                                    use_padding=False)
            elif self.backend == 'LSTM':
                output, _ = tfu.pass_static_lstm(self.cell, self.initial_state, self.k, res_vecs, scope)
                return tf.reshape(output, [self.lstm_hidden])

    def output_vector_size(self):
        if self.backend == 'SUM':
            return self.real_in_size
        elif self.backend == 'CNN':
            return self.num_filters * len(self.filter_sizes)
        elif self.backend == 'LSTM':
            return self.lstm_hidden

    def l2_loss(self):
        return tf.constant(0.0)