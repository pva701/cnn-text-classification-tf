import tensorflow as tf
import numpy as np


class TreeSimpleCNN(object):
    def init_binary_tree_simple(self
                                , num_classes
                                , vocab_size
                                , filter_sizes
                                , num_filters
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

            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.get_variable("W_conv_{}".format(filter_size), filter_shape,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                    b = tf.get_variable("b_conv_{}".format(filter_size), [num_filters],
                                        initializer=tf.constant_initializer(0.1))

            with tf.name_scope("recursive"):
                num_filters_total = num_filters * len(filter_sizes)
                in_size = num_filters_total
                out_size = num_filters_total

                W1 = tf.get_variable("W1",
                                     [num_filters_total, out_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
                W2 = tf.get_variable("W2",
                                     [num_filters_total, out_size],
                                     initializer=tf.contrib.layers.xavier_initializer())

                biases = tf.get_variable("biases_rec", [out_size],
                                         initializer=tf.zeros_initializer())


            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable("W",
                                    shape=[out_size, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases_out", [num_classes], initializer=tf.zeros_initializer())

    def __init__(self,
                 is_binary,
                 vocab_size,
                 filter_sizes,
                 num_filters,
                 embedding_size=None,
                 pretrained_embedding=None,
                 l2_reg_lambda=0.0):

        if is_binary:
            num_classes = 2
        else:
            num_classes = 5

        self.init_binary_tree_simple(num_classes, vocab_size,
                                     filter_sizes, num_filters,
                                     embedding_size, pretrained_embedding)
        if not embedding_size:
            embedding_size = pretrained_embedding.shape[1]

        self.words = tf.placeholder(tf.int32, [None], "words")  # n
        self.n_words = tf.placeholder(tf.int32, [], "n_words")  # n

        self.left = tf.placeholder(tf.int32, [None], "left")  # n - 1
        self.right = tf.placeholder(tf.int32, [None], "right")  # n - 1
        self.labels = tf.placeholder(tf.int32, [None, num_classes], "labels")  # 2n-1x5
        self.binary_ids = tf.placeholder(tf.int32, [None], "binary_ids")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("internal-state") as scope:
            scope.reuse_variables()
            with tf.name_scope("embedding"):
                embedding_matrix = tf.get_variable("embedding")
                embedded_vectors = tf.nn.embedding_lookup(embedding_matrix, self.words)
                expanded_vectors = tf.expand_dims(tf.expand_dims(embedded_vectors, 0), -1)
                # print(expanded_vectors.get_shape())

            conv_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                    W = tf.get_variable("W_conv_{}".format(filter_size))
                    b = tf.get_variable("b_conv_{}".format(filter_size))
                    padding = tf.zeros([1, filter_size - 1, embedding_size, 1])
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

            num_filters_total = num_filters * len(filter_sizes)
            conc_conv = tf.concat(conv_outputs, 3)
            flat_conv_unreg = tf.reshape(conc_conv, [-1, num_filters_total])

            # Add dropout
            with tf.name_scope("dropout"):
                flat_conv = tf.nn.dropout(flat_conv_unreg, self.dropout_keep_prob)

            with tf.name_scope("recursive"):
                W1 = tf.get_variable("W1")
                W2 = tf.get_variable("W2")
                biases_rec = tf.get_variable("biases_rec")

                def apply_children(vectors, i):
                    lc = vectors[self.left[i]]
                    rc = vectors[self.right[i]]
                    vector = tf.nn.sigmoid(tf.matmul(tf.expand_dims(lc, 0), W1) +
                                           tf.matmul(tf.expand_dims(rc, 0), W2) +
                                           biases_rec)
                    return tf.concat([vectors, vector], 0)

                hidden = tf.foldl(apply_children,
                                  tf.range(tf.constant(0), self.n_words - 1),
                                  initializer=flat_conv)

            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(hidden, self.dropout_keep_prob)

            # Calculate Mean cross-entropy loss
            with tf.name_scope("loss"):
                # Keeping track of l2 regularization loss (optional)
                l2_loss = tf.constant(0.0)

                W = tf.get_variable("W")
                biases_out = tf.get_variable("biases_out")

                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(biases_out)
                l2_loss += tf.nn.l2_loss(W1)
                l2_loss += tf.nn.l2_loss(W2)
                l2_loss += tf.nn.l2_loss(biases_rec)

                self.scores = tf.nn.xw_plus_b(h_drop, W, biases_out, name="scores")
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

                self.root_accuracy = tf.equal(self.predictions[-1],
                                              tf.argmax(self.labels[-1], axis=0),
                                              name="root_accuracy")
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
