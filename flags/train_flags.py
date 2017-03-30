__author__ = 'pva701'

import tensorflow as tf

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("dataset_word2vec_path", "./data/stanford_sentiment_treebank/word2vec.pickle"
                       , "Path to word2vec for current dataset")
tf.flags.DEFINE_string("word2vec_path", "./data/word2vec/GoogleNews-vectors-negative300.bin"
                       , "Path to word2vec")
tf.flags.DEFINE_string("sst_path", "/home/pva701/github/cnn-text-classification-tf/data/stanford_sentiment_treebank",
                       "Path to SST")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", None, "Dimensionality of character embedding, None for word2vec initialization of embedding")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0000, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 4000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()