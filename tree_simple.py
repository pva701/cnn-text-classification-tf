import tensorflow as tf
from tensorflow.contrib import rnn
from numpy import sqrt


class BinaryTreeSimple(object):
    def init_binary_tree_simple(self
                                , num_classes
                                , vocab_size
                                , embedding_size=None
                                , pretrained_embedding=None):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if embedding_size:
                _ = tf.get_variable("embedding",
                                    [vocab_size, embedding_size],
                                    tf.random_uniform_initializer)
            else:
                embedding_size = pretrained_embedding.shape[1]
                _ = tf.get_variable("embedding",
                                    [vocab_size, embedding_size],
                                    tf.constant_initializer(pretrained_embedding))

        with tf.name_scope("recursive"):
            W1 = tf.get_variable("W1",
                                 [embedding_size, embedding_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable("W2",
                                 [embedding_size, embedding_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            biases = tf.get_variable("biases_rec", [embedding_size],
                                     initializer=tf.constant_initializer(tf.zeros(embedding_size)))


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W",
                shape=[embedding_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("biases_out", initializer=tf.constant_initializer(tf.zeros(num_classes)))

        tf.get_variable_scope().reuse_variables()

    def __init__(self,
                 num_classes,
                 vocab_size,
                 embedding_size=None,
                 pretrained_embedding=None,
                 l2_reg_lambda=0.0):

        self.init_binary_tree_simple(num_classes, vocab_size, embedding_size, pretrained_embedding)
        hidden_size = embedding_size

        self.words = tf.placeholder(tf.int32, [None], "words")

        self.left = tf.placeholder(tf.int32, [None], "left")
        self.right = tf.placeholder(tf.int32, [None], "right")
        self.labels = tf.placeholder(tf.int32, [None], "labels")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding"):
            embedding_matrix = tf.get_variable("embedding")
            embedded_vectors = tf.nn.embedding_lookup(embedding_matrix, self.words)

        with tf.name_scope("recursive"):
            num_words = self.words.get_shape().as_list()[0]
            num_nodes = self.left.get_shape().as_list()[0]  # num_words - 1

            nodes_vectors = tf.Variable(tf.zeros([num_nodes, hidden_size]), "nodes_vectors")
            hidden_vectors_init = tf.concat([embedded_vectors, nodes_vectors], 0)

            W1 = tf.get_variable("W1")
            W2 = tf.get_variable("W2")
            biases = tf.get_variable("biases_rec")

            def apply_children(vectors, i):
                lc = vectors[self.left[i]]
                rc = vectors[self.right[i]]
                vectors[i] = tf.nn.sigmoid(tf.matmul(lc, W1) + tf.matmul(rc, W2) + biases)

            hidden = tf.foldl(apply_children, tf.range(tf.constant(0), num_nodes),
                                      initializer=hidden_vectors_init)

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

            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.scores)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def data_type(self):
        return tf.float32
        # return tf.float16 if FLAGS.use_fp16 else tf.float32
