__author__ = 'pva701'

import tensorflow as tf
import tf_utils as tfu

FUN_ACT = tf.sigmoid

class SubtreeTopK:
    def __init__(self, k, leaf_size=None, backend=None):
        self.k = k
        self.leaf_size = leaf_size
        self.backend = backend
        if self.backend != 'SUM' and self.backend != 'CNN' and self.backend != 'LSTM':
            raise Exception('Unexpected SubtreeInnerTopK backend')

    def init_with_scope(self, in_size):
        self.in_size = in_size
        with tf.variable_scope("subtree-top-k"):
            if self.leaf_size:
                tfu.linear(in_size, self.leaf_size, "top")
            if self.backend == 'SUM':
                tfu.linear(self.k, 1, "sum")

    def build_graph(self, n_words, words_vecs, _1, _2, _3, _4, euler, euler_l, euler_r, dropout_keep):
        with tf.variable_scope("subtree-top-k") as scope:
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
                top_k_vec = self.subtree_fn(nodes_vec)
                return tf.concat([vectors, top_k_vec], 0)

        return tf.foldl(apply_top_k,
                        tf.range(0, n_words - 1),
                        initializer=leaves_vectors)


    def subtree_fn(self, subtree_vecs):
        with tf.variable_scope("subtree-top-k") as scope:
            scope.reuse_variables()
            extended_subtree_vecs = tf.concat([subtree_vecs, tf.zeros([self.k, self.output_vector_size()])], 0)
            vec_len = tf.reduce_sum(tf.square(extended_subtree_vecs), 1)
            _, indices = tf.nn.top_k(vec_len, self.k, False, "top-vectors")
            res_vecs = tf.gather(extended_subtree_vecs, indices)
            if self.backend == 'SUM':
                W_sum = tf.get_variable("W_sum")
                biases_sum = tf.get_variable("biases_sum")
                return tf.transpose(FUN_ACT(tf.nn.xw_plus_b(tf.transpose(res_vecs), W_sum, biases_sum)))
            elif self.backend == 'CNN':
                raise Exception('CNN backend not implemented yet')
            elif self.backend == 'LSTM':
                raise Exception('LSTM backend not implemented yet')

    def output_vector_size(self):
        return self.leaf_size or self.in_size

    def l2_loss(self):
        return tf.constant(0.0)
