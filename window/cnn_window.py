__author__ = 'pva701'

import tensorflow as tf
import utils as tfu


class CnnWindow:
    def __init__(self, filter_sizes, num_filters, embedding_size, channels=1):
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.embedding_size = embedding_size
        self.num_filters_total = num_filters * len(filter_sizes)
        self.channels = channels

    def init_with_scope(self):
        with tf.variable_scope("cnn"):
            tfu.create_cnn(self.filter_sizes, self.num_filters, self.embedding_size)

    def build_graph(self, embedded_vectors, n_words, dropout_keep_prob):
        with tf.variable_scope("cnn") as scope:
            scope.reuse_variables()

            flat_conv_unreg = tfu.pass_cnn(
                self.filter_sizes,
                self.num_filters,
                embedded_vectors,
                self.embedding_size,
                use_padding=True)

        # Add dropout
        with tf.name_scope("dropout"):
            flat_conv = tf.nn.dropout(flat_conv_unreg, dropout_keep_prob, name="flat_conv")
        return flat_conv

    def output_vector_size(self):
        return self.num_filters_total

    def l2_loss(self):
        ret = tf.constant(0.0)
        with tf.variable_scope("cnn"):
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                    W = tf.get_variable("W_conv_{}".format(filter_size))
                    b = tf.get_variable("b_conv_{}".format(filter_size))
                    ret += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        return ret