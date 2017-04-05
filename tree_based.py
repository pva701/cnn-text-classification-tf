import tensorflow as tf
import numpy as np


class TreeBased(object):
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

        self.words = tf.placeholder(tf.int32, [None], "words")  # n
        self.n_words = tf.placeholder(tf.int32, [], "n_words")  # n

        self.left = tf.placeholder(tf.int32, [None], "left")  # n - 1
        self.right = tf.placeholder(tf.int32, [None], "right")  # n - 1
        self.labels = tf.placeholder(tf.int32, [None, num_classes], "labels")  # 2n-1x5
        self.binary_ids = tf.placeholder(tf.int32, [None], "binary_ids")
        self.weights_loss = tf.placeholder(tf.float32, [None], "weights_loss")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("internal-state") as scope:
            scope.reuse_variables()
            with tf.name_scope("embedding"):
                embedding_matrix = tf.get_variable("embedding")
                embedded_vectors = tf.nn.embedding_lookup(embedding_matrix, self.words)

            # Window here
            windows_repr = window_algo.build_graph(embedded_vectors, self.n_words, self.dropout_keep_prob)
            # Processing here
            out_repr_unstripped = processing_algo.build_graph(windows_repr, self.left, self.right, self.n_words, self.dropout_keep_prob)
            out_repr = out_repr_unstripped if not exclude_leaves_loss else out_repr_unstripped[self.n_words:]

            # Add dropout
            with tf.name_scope("dropout"):
                out_repr_drop = tf.nn.dropout(out_repr, self.dropout_keep_prob)

            # Calculate Mean cross-entropy loss
            with tf.name_scope("loss"):
                # Keeping track of l2 regularization loss (optional)
                l2_loss = tf.constant(0.0)

                W_out = tf.get_variable("W_out")
                biases_out = tf.get_variable("biases_out")

                l2_loss += tf.nn.l2_loss(W_out)
                l2_loss += tf.nn.l2_loss(biases_out)
                # l2_loss += tf.nn.l2_loss(W1)
                # l2_loss += tf.nn.l2_loss(W2)
                # l2_loss += tf.nn.l2_loss(biases_rec)

                self.scores = tf.nn.xw_plus_b(out_repr_drop, W_out, biases_out, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

                if is_binary:
                    losses = \
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=tf.gather(self.labels, self.binary_ids),
                            logits=tf.gather(self.scores, self.binary_ids),
                            name="losses")
                else:
                    losses = \
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=self.labels,
                            logits=self.scores,
                            name="losses")

                self.root_loss = \
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.labels[-1],
                        logits=self.scores[-1],
                        name="root_loss")

                if is_weight_loss:
                    self.loss = tf.reduce_sum(tf.multiply(self.weights_loss, losses)) + \
                                l2_reg_lambda * l2_loss
                else:
                    self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                if is_binary:
                    correct_predictions = \
                        tf.equal(
                            tf.gather(self.predictions, self.binary_ids),
                            tf.argmax(tf.gather(self.labels, self.binary_ids), 1))
                else:
                    correct_predictions = \
                        tf.equal(self.predictions, tf.argmax(self.labels, 1))

                self.root_accuracy = \
                    tf.equal(self.predictions[-1],
                             tf.argmax(self.labels[-1], axis=0),
                             name="root_accuracy")
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
