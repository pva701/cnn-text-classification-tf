import tensorflow as tf
import numpy as np


class BinaryTreeSimple(object):
    def init_binary_tree_simple(self
                                , num_classes
                                , vocab_size
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

            with tf.name_scope("recursive"):
                W1 = tf.get_variable("W1",
                                     [embedding_size, embedding_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
                W2 = tf.get_variable("W2",
                                     [embedding_size, embedding_size],
                                     initializer=tf.contrib.layers.xavier_initializer())

                biases = tf.get_variable("biases_rec", [embedding_size],
                                         initializer=tf.zeros_initializer())


            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable("W",
                                    shape=[embedding_size, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases_out", [num_classes], initializer=tf.zeros_initializer())

    def __init__(self,
                 num_classes,
                 vocab_size,
                 embedding_size=None,
                 pretrained_embedding=None,
                 l2_reg_lambda=0.0):
        self.init_binary_tree_simple(num_classes, vocab_size, embedding_size, pretrained_embedding)
        hidden_size = embedding_size

        self.words = tf.placeholder(tf.int32, [None], "words")  # n
        self.n_words = tf.placeholder(tf.int32, [], "n_words")  # n

        self.left = tf.placeholder(tf.int32, [None], "left")  # n - 1
        self.right = tf.placeholder(tf.int32, [None], "right")  # n - 1
        self.labels = tf.placeholder(tf.int32, [None, num_classes], "labels")  # 2n-1x5
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("internal-state") as scope:
            scope.reuse_variables()
            with tf.name_scope("embedding"):
                embedding_matrix = tf.get_variable("embedding")
                embedded_vectors = tf.nn.embedding_lookup(embedding_matrix, self.words)

            with tf.name_scope("recursive"):
                W1 = tf.get_variable("W1")
                W2 = tf.get_variable("W2")
                biases = tf.get_variable("biases_rec")

                def apply_children(vectors, i):
                    lc = vectors[self.left[i]]
                    rc = vectors[self.right[i]]
                    vector = tf.nn.sigmoid(tf.matmul(tf.expand_dims(lc, 0), W1) +
                                           tf.matmul(tf.expand_dims(rc, 0), W2) +
                                           biases)
                    return tf.concat([vectors, vector], 0)

                hidden = tf.foldl(apply_children,
                                  tf.range(tf.constant(0), self.n_words - 1),
                                      initializer=embedded_vectors)

            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(hidden, self.dropout_keep_prob)

            # Calculate Mean cross-entropy loss
            with tf.name_scope("loss"):
                # Keeping track of l2 regularization loss (optional)
                l2_loss = tf.constant(0.0)

                W = tf.get_variable("W")
                biases = tf.get_variable("biases_out")

                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(biases)

                self.scores = tf.nn.xw_plus_b(h_drop, W, biases, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

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

                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
                self.root_accuracy = tf.equal(self.predictions[-1],
                                              tf.argmax(self.labels[-1], axis=0),
                                              name="root_accuracy")
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def data_type(self):
        return tf.float32
        # return tf.float16 if FLAGS.use_fp16 else tf.float32
