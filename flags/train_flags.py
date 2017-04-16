__author__ = 'pva701'

import tensorflow as tf

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("dataset_embedding_path",
                       "./data/stanford_sentiment_treebank/word2vec.pickle",
                       #"./data/stanford_sentiment_treebank/glove.pickle",
                       "Path to word embedding for current dataset")
tf.flags.DEFINE_string("embedding_path",
                       #"./data/word2vec/GoogleNews-vectors-negative300.bin",
                       "./data/glove/glove.840B.300d.txt",
                       "Path to word embedding")
tf.flags.DEFINE_string("sst_path",
                       "./data/stanford_sentiment_treebank",
                       "Path to SST")

# Model Hyperparameters
tf.flags.DEFINE_boolean("is_binary", False, "Binary classification or fine-grained")
tf.flags.DEFINE_boolean("exclude_leaves_loss", False, "Exclude leaves loss from minimization")
tf.flags.DEFINE_boolean("weight_loss", False, "Weight loss (depends on length subtree)")

tf.flags.DEFINE_integer("embedding_dim", None,
                        "Dimensionality of character embedding, None for word2vec initialization of embedding")
tf.flags.DEFINE_string("window_algo", "CNN", "Specify window algo: CNN|LSTM|DUMMY")
tf.flags.DEFINE_string("processing_algo", "TREE-LSTM", "Specify processing algo: SIMPLE|TREE-LSTM")
tf.flags.DEFINE_integer("mem_size", 150,
                        "Size of memory and hidden state (for TREE-LSTM")

tf.flags.DEFINE_integer("recursive_size", None,
                        "Size of sentiment vectors (for SIMPLE processing algo)")
tf.flags.DEFINE_integer("lstm_hidden", 200, "LSTM hidden size, when use LSTM (for LSTM window)")
tf.flags.DEFINE_string("filter_sizes", "2, 3, 4",
                       "Comma-separated filter sizes (default: '2,3,4') (for CNN window)")
tf.flags.DEFINE_integer("num_filters", 128,
                        "Number of filters per filter size (default: 128) (for CNN window)")

tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001,
                      "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_string("optimizer", "Adagrad", "Optimizer")
tf.flags.DEFINE_integer("minibatch", 25, "Minibatch size")
tf.flags.DEFINE_integer("num_epochs", 2000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 80, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 160, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()