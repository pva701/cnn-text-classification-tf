__author__ = 'pva701'

import tensorflow as tf


def linear(rows, cols, name):
    W = tf.get_variable(
        "W_" + name,
        [rows, cols],
        initializer=tf.contrib.layers.xavier_initializer())
    biases_top = tf.get_variable(
        "biases_" + name,
        [cols],
        initializer=tf.zeros_initializer())
