import tensorflow as tf
import numpy as np
import utils as tfu

class TreeBased:
    def init_tree_based(self,
                        num_classes,
                        vocab_size,
                        window_algo,
                        processing_algo,
                        outer_algo,
                        embedding_size=None,
                        pretrained_embedding=None):
        with tf.variable_scope("internal-state"):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                if embedding_size:
                    _ = tf.get_variable("embedding",
                                        # [vocab_size, embedding_size],
                                        initializer=tf.random_uniform_initializer)
                else:
                    embedding_size = pretrained_embedding.shape[1]
                    _ = tf.get_variable("embedding",
                                        shape=[vocab_size, embedding_size],
                                        initializer=tf.constant_initializer(pretrained_embedding, dtype=np.float32))

            window_algo.init_with_scope()
            window_vector_size = window_algo.output_vector_size()
            processing_algo.init_with_scope(window_vector_size)
            processed_vector_size = processing_algo.output_vector_size()
            outer_algo.init_with_scope(window_vector_size, processed_vector_size)
            outer_vector_size = outer_algo.output_vector_size()

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                tfu.linear(processed_vector_size + outer_vector_size, num_classes, "out")

    def __init__(self,
                 is_binary,
                 vocab_size,
                 window_algo, processing_algo, outer_algo,
                 exclude_leaves_loss,
                 embedding_size=None,
                 pretrained_embedding=None,
                 l2_reg_lambda=0.0):

        if is_binary:
            num_classes = 2
        else:
            num_classes = 5

        self.init_tree_based(num_classes, vocab_size,
                             window_algo, processing_algo, outer_algo,
                             embedding_size, pretrained_embedding)

        self.window_algo = window_algo
        self.processing_algo = processing_algo
        self.exclude_leaves_loss = exclude_leaves_loss
        self.is_binary = is_binary
        self.l2_reg_lambda = l2_reg_lambda
        self.outer_algo = outer_algo

    def init_before_minibatch(self, w, nw, l, r, lbound, rbound, lb,
                              euler, euler_l, euler_r, bi, d):
        self.words, \
        self.n_words, \
        self.left, \
        self.right, \
        self.l_bound, \
        self.r_bound, \
        self.labels, \
        self.euler, \
        self.euler_l, \
        self.euler_r, \
        self.binary_ids, \
        self.dropout_keep_prob = \
            w, nw, l, r, lbound, rbound, lb, euler, euler_l, euler_r, bi, d

    def fn(self, i):
        n_words = self.n_words[i]
        lb_size = n_words - 1 if self.exclude_leaves_loss else 2 * n_words - 1
        words = self.words[i][:n_words]
        left = self.left[i][:n_words]
        right = self.right[i][:n_words]
        l_bound, r_bound = self.l_bound[i], self.r_bound[i]
        labels = self.labels[i][:lb_size]

        euler = self.euler[i][:2 * n_words - 1]
        euler_l = self.euler_l[i][:2 * n_words - 1]
        euler_r = self.euler_r[i][:2 * n_words - 1]
        binary_ids = None
        # binary_ids = self.binary_ids[i][:n_words]

        with tf.variable_scope("internal-state") as scope:
            scope.reuse_variables()
            with tf.name_scope("embedding"):
                embedding_matrix = tf.get_variable("embedding")
                embedded_vectors = tf.nn.embedding_lookup(embedding_matrix, words)

            # Window here
            windows_repr = self.window_algo.build_graph(embedded_vectors, n_words, self.dropout_keep_prob)

            # Processing here
            raw_out_repr = \
                self.processing_algo.build_graph(
                    n_words, windows_repr, left, right,
                    l_bound, r_bound,
                    euler, euler_l, euler_r,
                    self.dropout_keep_prob)

            if self.outer_algo:
                outer_vectors = self.outer_algo.build_graph(
                        raw_out_repr, windows_repr, n_words, left, right,
                        l_bound, r_bound,
                        euler, euler_l, euler_r,
                        self.dropout_keep_prob)

                out_repr_with_leaves = tf.concat([raw_out_repr, outer_vectors], 1)
            else:
                out_repr_with_leaves = raw_out_repr

            if not self.exclude_leaves_loss:
                out_repr = out_repr_with_leaves
            else:
                out_repr = out_repr_with_leaves[n_words:]

            return self.__build_loss_accuracy(out_repr, labels, binary_ids)

    def __build_loss_accuracy(self, vrs, labels, binary_ids):
        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            W_out = tf.get_variable("W_out")
            biases_out = tf.get_variable("biases_out")

            scores = tf.nn.xw_plus_b(vrs, W_out, biases_out, name="scores")
            predictions = tf.argmax(scores, 1, name="predictions")

            if self.is_binary:
                losses = \
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.gather(labels, binary_ids),
                        logits=tf.gather(scores, binary_ids),
                        name="losses")
            else:
                losses = \
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels,
                        logits=scores,
                        name="losses")

            root_loss = \
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels[-1],
                    logits=scores[-1],
                    name="root_loss")

            loss = tf.reduce_sum(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            if self.is_binary:
                correct_predictions = \
                    tf.equal(
                        tf.gather(predictions, binary_ids),
                        tf.argmax(tf.gather(labels, binary_ids), 1))
            else:
                correct_predictions = \
                    tf.equal(predictions, tf.argmax(labels, 1))

            root_accuracy = \
                tf.cast(tf.equal(predictions[-1],
                                 tf.argmax(labels[-1], axis=0),
                                 name="root_accuracy"),
                        "float")
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        return tf.stack([loss, root_loss, accuracy, root_accuracy])

    def l2_loss(self):
        with tf.variable_scope("internal-state") as scope:
            scope.reuse_variables()
            W_out = tf.get_variable("W_out")
            biases_out = tf.get_variable("biases_out")

            with tf.name_scope("l2-reg"):
                # Keeping track of l2 regularization loss (optional)
                ret = tf.constant(0.0)
                ret += tf.nn.l2_loss(W_out) + tf.nn.l2_loss(biases_out)
                ret += self.window_algo.l2_loss()
                ret += self.processing_algo.l2_loss()

                return self.l2_reg_lambda * ret
