__author__ = 'pva701'

import tensorflow as tf
import tf_utils as tfu

FUN_ACT = tf.sigmoid


class SubtreeTopK:
    def __init__(self, k, leaf_size=None, backend=None, num_filters=None, filter_sizes=None):
        self.k = k
        self.leaf_size = leaf_size
        self.backend = backend
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        if self.backend != 'SUM' and self.backend != 'CNN' and self.backend != 'LSTM':
            raise Exception('Unexpected SubtreeInnerTopK backend')

    def init_with_scope(self, in_size):
        self.in_size = in_size
        with tf.variable_scope("subtree-top-k"):
            self.real_in_size = self.leaf_size or self.in_size

            if self.leaf_size:
                tfu.linear(in_size, self.leaf_size, "top")
            if self.backend == 'SUM':
                tfu.linear(self.k, 1, "sum")
            elif self.backend == 'CNN':
                if self.filter_sizes is None:
                    self.filter_sizes = [self.k]

                for i, filter_size in enumerate(self.filter_sizes):
                    with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                        # Convolution Layer
                        filter_shape = [filter_size, self.real_in_size, 1, self.num_filters]
                        W = tf.get_variable("W_conv_{}".format(filter_size), filter_shape,
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                        b = tf.get_variable("b_conv_{}".format(filter_size), [self.num_filters],
                                            initializer=tf.constant_initializer(0.1))

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
            extended_subtree_vecs = tf.concat([subtree_vecs, tf.zeros([self.k, self.real_in_size])], 0)
            vec_len = tf.reduce_sum(tf.square(extended_subtree_vecs), 1)
            _, indices = tf.nn.top_k(vec_len, self.k, False, "top-vectors")
            res_vecs = tf.gather(extended_subtree_vecs, indices)
            if self.backend == 'SUM':
                W_sum = tf.get_variable("W_sum")
                biases_sum = tf.get_variable("biases_sum")
                return tf.transpose(FUN_ACT(tf.nn.xw_plus_b(tf.transpose(res_vecs), W_sum, biases_sum)))
            elif self.backend == 'CNN':
                pooled_outputs = []
                for i, filter_size in enumerate(self.filter_sizes):
                    with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                        W = tf.get_variable("W_conv_{}".format(filter_size))
                        b = tf.get_variable("b_conv_{}".format(filter_size))
                        conv = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(res_vecs, 0), -1),
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                        # Maxpooling over the outputs
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, self.k - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
                        pooled_outputs.append(pooled)

                num_filters_total = self.num_filters * len(self.filter_sizes)
                conc_pooled = tf.concat(pooled_outputs, 3)
                pooled_flat = tf.reshape(conc_pooled, [-1, num_filters_total])
                return pooled_flat
            elif self.backend == 'LSTM':
                raise Exception('LSTM backend not implemented yet')

    def output_vector_size(self):
        if self.backend == 'SUM':
            return self.leaf_size or self.in_size
        elif self.backend == 'CNN':
            return self.num_filters * len(self.filter_sizes)

    def l2_loss(self):
        return tf.constant(0.0)
