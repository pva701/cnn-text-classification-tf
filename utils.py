__author__ = 'pva701'

import tensorflow as tf

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


# Git commit

import subprocess

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()