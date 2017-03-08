import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


class LstmOverLstm(object):
    def __init__(self
                 , sequence_length
                 , num_classes, vocab_size
                 , lstm_parameters=[(128, 5)]
                 , lstm_strides=None
                 , embedding_size=None
                 , pretrained_embedding=None
                 , l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.batch_size_ = tf.placeholder(tf.int32, name="batch_size")
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if embedding_size:
                W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
            else:
                # embedding_size = pretrained_embedding.shape[1]
                W = tf.Variable(
                    tf.constant(pretrained_embedding, dtype=np.float32),
                    name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        cells = []
        self.initial_states = []
        layers = len(lstm_parameters)
        for lstm_size, _ in lstm_parameters:
            cell = rnn.BasicLSTMCell(lstm_size)
            cells.append(cell)
            self.initial_states.append(cell.zero_state(self.batch_size_, self.data_type()))

        prev_sequence = self.embedded_chars
        with tf.variable_scope("LSTM"):
            for layer in range(layers):
                with tf.variable_scope("lstm-layer-{}".format(layer)):
                    cell_outputs = []
                    cell = cells[layer]
                    prev_seq_length = prev_sequence.get_shape().as_list()[1]
                    (lstm_size, lstm_window) = lstm_parameters[layer]
                    lstm_stride = 1 if lstm_strides is None \
                                       or len(lstm_strides) != layers \
                                       or lstm_strides[layer] is None \
                                    else lstm_strides[layer]

                    print("Layer #{}".format(layer))
                    print("Hidden size = {}, window size = {}, stride = {}".format(lstm_size, lstm_window, lstm_stride))
                    for start_position in range(0, prev_seq_length, lstm_stride):
                        print("Start position {}".format(start_position))
                        state = self.initial_states[layer]
                        output = None
                        rbound = min(start_position + lstm_window, prev_seq_length)
                        for i in range(start_position, rbound):
                            if i > 0:
                                tf.get_variable_scope().reuse_variables()
                            output, state = cell(prev_sequence[:, i, :], state)
                        cell_outputs.append(output)

                    prev_sequence = tf.stack(cell_outputs, 1)
                    print("Prev sequence shape", prev_sequence.shape)

        lstm_output = tf.expand_dims(prev_sequence, -1)
        _, out_length, out_dim, _ = lstm_output.get_shape().as_list()
        print(lstm_output.shape)

        #Max pool part
        with tf.name_scope("maxpool"):
            # Apply nonlinearity
            h = tf.nn.relu(lstm_output, name="relu")
            # Maxpooling over the outputs
            self.pooled = tf.nn.max_pool(
                h,
                ksize=[1, out_length, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            print(self.pooled.shape)
            self.pooled_resize = tf.reshape(self.pooled, [-1, out_dim])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.pooled_resize, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[out_dim, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def data_type(self):
        return tf.float32
        # return tf.float16 if FLAGS.use_fp16 else tf.float32
