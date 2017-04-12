__author__ = 'pva701'
import tensorflow as tf


class MinibatchOptimizer:
    def __init__(self, optimizer, global_step, what, num_classes):
        self.batch_size = tf.placeholder(tf.int32, [], "batch_size")  # n
        self.words = tf.placeholder(tf.int32, [None, None], "words")  # n
        self.n_words = tf.placeholder(tf.int32, [None], "n_words")  # n

        self.left = tf.placeholder(tf.int32, [None, None], "left")  # n - 1
        self.right = tf.placeholder(tf.int32, [None, None], "right")  # n - 1
        self.l_bound = tf.placeholder(tf.int32, [None, None], "l_bound")  # n - 1
        self.r_bound = tf.placeholder(tf.int32, [None, None], "r_bound")  # n - 1

        self.labels = tf.placeholder(tf.int32, [None, None, num_classes], "labels")  # 2n-1x5
        # self.binary_ids = tf.placeholder(tf.int32, [None, None], "binary_ids")
        self.weights_loss = tf.placeholder(tf.float32, [None, None], "weights_loss")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        what. \
            init_before_minibatch(
            self.words, self.n_words, self.left, self.right, self.l_bound, self.r_bound,
            self.labels, None, self.weights_loss, self.dropout_keep_prob)

        self.result = tf.reduce_mean(
            tf.map_fn(
                lambda i: what.fn(i),
                tf.range(tf.constant(0), self.batch_size),
                dtype=tf.float32)
            , 0)
        loss = self.result[0] + what.l2_loss()
        grads_and_vars = optimizer.compute_gradients(loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars,
                                                  global_step=global_step)
