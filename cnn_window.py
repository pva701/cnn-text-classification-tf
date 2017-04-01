__author__ = 'pva701'

import tensorflow as tf


class CnnWindow:
    def __init__(self, filter_sizes, num_filters, embedding_size, channels=1):
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.embedding_size = embedding_size
        self.num_filters_total = num_filters * len(filter_sizes)
        self.channels = channels

    def init_with_scope(self):
        with tf.variable_scope("internal-state"):
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_size, self.channels, self.num_filters]
                    W = tf.get_variable("W_conv_{}".format(filter_size), filter_shape,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                    b = tf.get_variable("b_conv_{}".format(filter_size), [self.num_filters],
                                        initializer=tf.constant_initializer(0.1))

    def build_graph(self, embedded_vectors, n_words, dropout_keep_prob):
        expanded_vectors = tf.expand_dims(tf.expand_dims(embedded_vectors, 0), -1)
        conv_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                W = tf.get_variable("W_conv_{}".format(filter_size))
                b = tf.get_variable("b_conv_{}".format(filter_size))
                padding = tf.zeros([1, filter_size - 1, self.embedding_size, 1])
                conv = tf.nn.conv2d(
                    tf.concat([expanded_vectors, padding], 1),
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # print("CONV: ", conv.get_shape())
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # print("H: ", h.get_shape())
                conv_outputs.append(h)

        conc_conv = tf.concat(conv_outputs, 3)
        flat_conv_unreg = tf.reshape(conc_conv, [-1, self.num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            flat_conv = tf.nn.dropout(flat_conv_unreg, dropout_keep_prob, name="flat_conv")
        return flat_conv

    def output_vector_size(self):
        return self.num_filters_total
