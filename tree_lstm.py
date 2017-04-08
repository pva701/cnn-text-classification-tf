__author__ = 'pva701'

import tensorflow as tf


class TreeLstm:
    def __init__(self, mem_size):
        self._params_names = []
        self.mem_size = mem_size

    def _create_linear(self, name, in_size, out_size):
        self._params_names.append("W_" + name)
        self._params_names.append("b_" + name)
        tf.get_variable("W_" + name, [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        tf.get_variable("b_" + name, [out_size], initializer=tf.zeros_initializer())

    def init_with_scope(self, in_size):
        mem_size = self.mem_size
        with tf.variable_scope("tree-lstm"):
            with tf.name_scope("leaf"):
                self._create_linear("leaf_c", in_size, mem_size)
                self._create_linear("leaf_o", in_size, mem_size)

            with tf.name_scope("inner"):
                self._create_linear("i_l", mem_size, mem_size)
                self._create_linear("i_r", mem_size, mem_size)

                self._create_linear("f_l_l", mem_size, mem_size)
                self._create_linear("f_l_r", mem_size, mem_size)
                self._create_linear("f_r_l", mem_size, mem_size)
                self._create_linear("f_r_r", mem_size, mem_size)

                self._create_linear("o_l", mem_size, mem_size)
                self._create_linear("o_r", mem_size, mem_size)

                self._create_linear("u_l", mem_size, mem_size)
                self._create_linear("u_r", mem_size, mem_size)

    def _gate(self, name, lh, rh):
        W_l = tf.get_variable("W_" + name + "_l")
        b_l = tf.get_variable("b_" + name + "_l")

        W_r = tf.get_variable("W_" + name + "_r")
        b_r = tf.get_variable("b_" + name + "_r")
        return tf.matmul(tf.expand_dims(lh, 0), W_l) + b_l + \
               tf.matmul(tf.expand_dims(rh, 0), W_r) + b_r

    def build_graph(self, inputs, left, right, n_words, dropout_keep_prob):
        with tf.variable_scope("tree-lstm") as scope:
            scope.reuse_variables()

            with tf.name_scope("leaf"):
                W_leaf_c = tf.get_variable("W_leaf_c")
                b_leaf_c = tf.get_variable("b_leaf_c")
                W_leaf_o = tf.get_variable("W_leaf_o")
                b_leaf_o = tf.get_variable("b_leaf_o")

                c_leaf = tf.matmul(inputs, W_leaf_c) + b_leaf_c
                o_leaf = tf.sigmoid(tf.matmul(inputs, W_leaf_o) + b_leaf_o)
                h_leaf = tf.multiply(o_leaf, tf.tanh(c_leaf))

            with tf.name_scope("inner"):
                def compute_hidden_and_memory(state, index):
                    h_mat = state[0]
                    c_mat = state[1]
                    li, ri = left[index], right[index]

                    i_gate = tf.sigmoid(self._gate("i", h_mat[li], h_mat[ri]))
                    f_l_gate = tf.sigmoid(self._gate("f_l", h_mat[li], h_mat[ri]))
                    f_r_gate = tf.sigmoid(self._gate("f_r", h_mat[li], h_mat[ri]))
                    o_gate = tf.sigmoid(self._gate("o", h_mat[li], h_mat[ri]))
                    u = tf.tanh(self._gate("u", h_mat[li], h_mat[ri]))

                    c = tf.multiply(i_gate, u) + \
                        tf.multiply(f_l_gate, c_mat[li]) + \
                        tf.multiply(f_r_gate, c_mat[ri])

                    h = tf.multiply(o_gate, tf.tanh(c))

                    return tf.stack([tf.concat([h_mat, h], 0), tf.concat([c_mat, c], 0)])

                ret = tf.foldl(compute_hidden_and_memory,
                               tf.range(tf.constant(0), n_words - 1),
                               initializer=(h_leaf, c_leaf))[0]
                # Add dropout
                with tf.name_scope("dropout"):
                    return tf.nn.dropout(ret, dropout_keep_prob)

    def output_vector_size(self):
        return self.mem_size

    def l2_loss(self):
        ret = tf.constant(0.0)
        with tf.variable_scope("tree-lstm") as scope:
            scope.reuse_variables()

            for name in self._params_names:
                ret += tf.nn.l2_loss(tf.get_variable(name))
        return ret
