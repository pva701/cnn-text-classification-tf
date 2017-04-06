import tensorflow as tf
import numpy as np


class TreeBased:
    def init_binary_tree_simple(self
                                , num_classes
                                , vocab_size
                                , window_algo, processing_algo
                                , embedding_size=None
                                , pretrained_embedding=None):
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

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable("W_out",
                                    shape=[processed_vector_size, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases_out", [num_classes], initializer=tf.zeros_initializer())

    def __init__(self,
                 is_binary,
                 vocab_size,
                 window_algo, processing_algo,
                 exclude_leaves_loss,
                 is_weight_loss,
                 embedding_size=None,
                 pretrained_embedding=None,
                 l2_reg_lambda=0.0):

        if is_binary:
            num_classes = 2
        else:
            num_classes = 5

        self.init_binary_tree_simple(num_classes, vocab_size,
                                     window_algo, processing_algo,
                                     embedding_size, pretrained_embedding)

        self.window_algo = window_algo
        self.processing_algo = processing_algo
        self.exclude_leaves_loss = exclude_leaves_loss
        self.is_binary = is_binary
        self.is_weight_loss = is_weight_loss
        self.l2_reg_lambda = l2_reg_lambda

    def init_before_minibatch(self, w, nw, l, r, lb, bi, wl, d):
        self.words, self.n_words, self.left, self.right, self.labels, self.binary_ids, self.weights_loss, self.dropout_keep_prob = \
            w, nw, l, r, lb, bi, wl, d

    def fn(self, i):
        n_words = self.n_words[i]
        lb_size = n_words - 1 if self.exclude_leaves_loss else 2 * n_words - 1
        words = self.words[i][:n_words]
        left = self.left[i][:n_words]
        right = self.right[i][:n_words]
        labels = self.labels[i][:lb_size]
        #binary_ids = self.binary_ids[i][:n_words]
        weights_loss = self.weights_loss[i][:lb_size]

        with tf.variable_scope("internal-state") as scope:
            scope.reuse_variables()
            with tf.name_scope("embedding"):
                embedding_matrix = tf.get_variable("embedding")
                embedded_vectors = tf.nn.embedding_lookup(embedding_matrix, words)

            # Window here
            windows_repr = self.window_algo.build_graph(embedded_vectors, n_words, self.dropout_keep_prob)

            # Processing here
            out_repr_unstripped = self.processing_algo.build_graph(windows_repr, left, right, n_words, self.dropout_keep_prob)
            out_repr = out_repr_unstripped if not self.exclude_leaves_loss else out_repr_unstripped[n_words:]

            # Calculate Mean cross-entropy loss
            with tf.name_scope("loss"):
                # Keeping track of l2 regularization loss (optional)
                l2_loss = tf.constant(0.0)

                W_out = tf.get_variable("W_out")
                biases_out = tf.get_variable("biases_out")

                l2_loss += tf.nn.l2_loss(W_out) + tf.nn.l2_loss(biases_out)
                l2_loss += self.window_algo.l2_loss()
                l2_loss += self.processing_algo.l2_loss()

                scores = tf.nn.xw_plus_b(out_repr, W_out, biases_out, name="scores")
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

                if self.is_weight_loss:
                    loss = tf.reduce_sum(tf.multiply(weights_loss, losses))
                else:
                    loss = tf.reduce_mean(losses)
                loss += self.l2_reg_lambda * l2_loss

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
