__author__ = 'pva701'

import tensorflow as tf
from tensorflow.contrib import rnn


# Tensor flow utils
def linear(rows, cols, name):
    W = tf.get_variable(
        "W_" + name,
        [rows, cols],
        initializer=tf.contrib.layers.xavier_initializer())
    biases_top = tf.get_variable(
        "biases_" + name,
        [cols],
        initializer=tf.zeros_initializer())


def create_lstm(hidden_size, in_size, scope):
    cell = rnn.BasicLSTMCell(hidden_size)
    initial_state = cell.zero_state(1, tf.float32)
    cell(tf.zeros([1, in_size]), initial_state, scope)
    return cell, initial_state


def pass_static_lstm(cell, init_state, len, vecs, scope):
    output = None
    state = init_state
    for i in range(len):
        output, state = cell(tf.expand_dims(vecs[i], 0), state, scope)
    return output, state


def create_cnn(filter_sizes, num_filters, cols):
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-{}".format(filter_size)):
            # Convolution Layer
            filter_shape = [filter_size, cols, 1, num_filters]
            tf.get_variable("W_conv_{}".format(filter_size), filter_shape,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            tf.get_variable("b_conv_{}".format(filter_size), [num_filters],
                            initializer=tf.constant_initializer(0.1))


def pass_cnn(filter_sizes, num_filters, vecs, cols, seq_len=None, use_padding=False):
    outputs = []
    expanded_vectors = tf.expand_dims(tf.expand_dims(vecs, 0), -1)
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-{}".format(filter_size)):
            W = tf.get_variable("W_conv_{}".format(filter_size))
            b = tf.get_variable("b_conv_{}".format(filter_size))
            if use_padding:
                padding = tf.zeros([1, filter_size - 1, cols, 1])
                padded_vectors = tf.concat([expanded_vectors, padding], 1)
            else:
                padded_vectors = expanded_vectors

            conv = tf.nn.conv2d(padded_vectors,
                                W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            if seq_len:
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                outputs.append(pooled)
            else:
                outputs.append(h)

    num_filters_total = num_filters * len(filter_sizes)
    concated = tf.concat(outputs, 3)
    if seq_len:
        return tf.reshape(concated, [num_filters_total])
    else:
        return tf.reshape(concated, [-1, num_filters_total])

# Git commit

import subprocess


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
