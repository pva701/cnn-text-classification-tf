__author__ = 'pva701'
from tensorflow.contrib import rnn
import tensorflow as tf

class LstmWindow:


    def __init__(self, hidden_size, embedded_size, min_window_size=2, max_window_size=4):
        self.hidden_size = hidden_size
        self.window_bounds = (min_window_size, max_window_size)
        self.embedded_size = embedded_size
        self.out_size = (max_window_size - min_window_size + 1) * hidden_size

    def init_with_scope(self):
        with tf.variable_scope("LSTM") as scope:
            self.cell = rnn.BasicLSTMCell(self.hidden_size)
            self.initial_state = self.cell.zero_state(1, tf.float32)
            state = self.initial_state
            self.cell(tf.zeros([1, self.embedded_size]), state, scope)

    def build_graph(self, embedded_vectors, n_words, dropout_keep_prob):
        min_size = self.window_bounds[0]
        max_size = self.window_bounds[1]

        with tf.variable_scope("LSTM") as scope:
            scope.reuse_variables()

            padding = tf.zeros([max_size - 1, self.embedded_size])
            ext_vector = tf.concat([embedded_vectors, padding], 0)

            def compute_vector(start_position):
                state = self.initial_state
                cell_outputs = []
                for i in range(max_size):
                    output, state = self.cell(tf.expand_dims(ext_vector[start_position + i], 0),
                                              state,
                                              scope)
                    if i + 1 >= min_size:
                        cell_outputs.append(output)

                return tf.reshape(tf.concat(cell_outputs, 1), [-1])

            ret = tf.map_fn(compute_vector, tf.range(tf.constant(0), n_words), dtype=tf.float32)
            # Add dropout
            with tf.name_scope("dropout"):
                return tf.nn.dropout(ret, dropout_keep_prob)

    def output_vector_size(self):
        return self.out_size

    def l2_loss(self):
        ret = tf.constant(0.0)
        with tf.variable_scope("LSTM") as scope:
            scope.reuse_variables()
            for var in tf.trainable_variables():
                if "LSTM/" in var.name:
                    ret += tf.nn.l2_loss(var)
        return ret
