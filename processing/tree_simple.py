__author__ = 'pva701'

import tensorflow as tf


class TreeSimple:
    def init_with_scope(self, in_size):
        self.in_size = in_size
        with tf.variable_scope("tree-simple") as scope:
            if not self.recursive_size:
                recursive_size = in_size
            else:
                W_t = tf.get_variable(
                    "W_t",
                    [in_size, self.recursive_size],
                    initializer=tf.contrib.layers.xavier_initializer())

                biases_t = tf.get_variable(
                    "biases_t",
                    [self.recursive_size],
                    initializer=tf.zeros_initializer())
                recursive_size = self.recursive_size

            W1_rec = tf.get_variable(
                "W1_rec",
                [recursive_size, recursive_size],
                initializer=tf.contrib.layers.xavier_initializer())
            W2_rec = tf.get_variable(
                "W2_rec",
                [recursive_size, recursive_size],
                initializer=tf.contrib.layers.xavier_initializer())

            biases = tf.get_variable(
                "biases_rec",
                [recursive_size],
                initializer=tf.zeros_initializer())
            self.subtree_fun.init_with_scope(self.recursive_size or self.in_size, in_size)

    def __init__(self, recursive_size, subtree_fun=None):
        self.recursive_size = recursive_size
        self.subtree_fun = subtree_fun

    def build_graph(self,
                    n_words,
                    words_vecs,
                    left, right,
                    l_bound, r_bound,
                    _1, _2, _3,
                    dropout_keep_prob):
        with tf.variable_scope("tree-simple") as scope:
            scope.reuse_variables()

            if self.recursive_size:
                W_t = tf.get_variable("W_t")
                biases_t = tf.get_variable("biases_t")
                # nn.tanh()
                leaves_vectors = tf.nn.sigmoid(tf.matmul(words_vecs, W_t) + biases_t)
            else:
                leaves_vectors = words_vecs

            W1 = tf.get_variable("W1_rec")
            W2 = tf.get_variable("W2_rec")
            biases_rec = tf.get_variable("biases_rec")

            def apply_children(vectors, i):
                lc = vectors[left[i]]
                rc = vectors[right[i]]
                # nn.tanh()
                vector = tf.nn.sigmoid(tf.matmul(tf.expand_dims(lc, 0), W1) +
                                       tf.matmul(tf.expand_dims(rc, 0), W2) +
                                       biases_rec)
                return tf.concat([vectors, vector], 0)

            raw_vectors = tf.foldl(apply_children,
                           tf.range(tf.constant(0), n_words - 1),
                           initializer=leaves_vectors)

            if self.subtree_fun:
                def apply_subtree_fun(i):
                    vector = raw_vectors[i + n_words]
                    l_b = l_bound[i]
                    r_b = r_bound[i]
                    return self.subtree_fun.fn(vector, words_vecs[l_b:r_b], r_b - l_b, dropout_keep_prob)


                inner_vectors = tf.map_fn(
                    apply_subtree_fun,
                    tf.range(tf.constant(0), n_words - 1),
                    dtype=tf.float32)
                ret_vectors = tf.concat([raw_vectors[:n_words], inner_vectors], 0)
            else:
                ret_vectors = raw_vectors

            # Add dropout
            with tf.name_scope("dropout"):
                return tf.nn.dropout(ret_vectors, dropout_keep_prob)

    def output_vector_size(self):
        if self.subtree_fun:
            return self.subtree_fun.output_vector_size()

        if not self.recursive_size:
            return self.in_size
        return self.recursive_size

    def l2_loss(self):
        with tf.variable_scope("tree-simple") as scope:
            scope.reuse_variables()
            ret = tf.constant(0.0)
            if self.recursive_size:
                ret += tf.nn.l2_loss(tf.get_variable("W_t"))
                ret += tf.nn.l2_loss(tf.get_variable("biases_t"))
            ret += tf.nn.l2_loss(tf.get_variable("W1_rec"))
            ret += tf.nn.l2_loss(tf.get_variable("W2_rec"))
            ret += tf.nn.l2_loss(tf.get_variable("biases_rec"))
            return ret