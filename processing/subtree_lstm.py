__author__ = 'pva701'

import tensorflow as tf
from tensorflow.contrib import rnn


class SubtreeLstm:
    def init_with_scope(self, in_size):
        self.in_size = in_size
        with tf.variable_scope("subtree-lstm") as scope:
            self.cell = rnn.BasicLSTMCell(self.hidden_size)
            self.initial_state = self.cell.zero_state(1, tf.float32)
            state = self.initial_state
            self.cell(tf.zeros([1, in_size]), state, scope)

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def fn(self, init_state, word_vecs, sub_n_words, dropout_keep_prob):
        with tf.variable_scope("subtree-lstm") as scope:
            scope.reuse_variables()

            def apply_children(state, i):
                _, new_state = self.cell(tf.expand_dims(word_vecs[i], 0),
                                         rnn.LSTMStateTuple(state[0], state[1]),
                                         scope)
                c, h = new_state
                return tf.stack([c, h])
            ret = tf.foldl(apply_children,
                           tf.range(tf.constant(0), sub_n_words),
                           initializer=(tf.zeros([1, self.hidden_size]), tf.expand_dims(init_state, 0)))
            # Add dropout
            with tf.name_scope("dropout"):
                return tf.reshape(tf.nn.dropout(ret[1], dropout_keep_prob), [-1])

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