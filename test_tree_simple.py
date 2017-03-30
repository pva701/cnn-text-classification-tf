#! /usr/bin/env python

import os

import tensorflow as tf
from tensorflow.contrib import learn

import pytreebank

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("sst_path", "./data/stanford_sentiment_treebank", "SST dataset path")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1490879569/checkpoints/",
                       "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


print("Loading data..")
dataset = pytreebank.load_sst(FLAGS.sst_path)
x_test = [x.lowercase() for x in dataset["test"]]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
vocab_dict = vocab_processor.vocabulary_._mapping
vocab_size = len(vocab_dict)
print("Vocab size: {}".format(vocab_size))

print("\nEvaluating...\n")
# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        words = graph.get_operation_by_name("words").outputs[0]
        n_words = graph.get_operation_by_name("n_words").outputs[0]
        left = graph.get_operation_by_name("left").outputs[0]
        right = graph.get_operation_by_name("right").outputs[0]
        labels = graph.get_operation_by_name("labels").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        #predictions = graph.get_operation_by_name("loss/predictions").outputs[0]
        accuracy = graph.get_operation_by_name("internal-state_1/accuracy/accuracy").outputs[0]
        root_accuracy = graph.get_operation_by_name("internal-state_1/accuracy/root_accuracy").outputs[0]

        test_examples = len(x_test)
        sum_acc = 0.0
        sum_root_acc = 0.0
        for ex in x_test:
            x, x_left, x_right, x_labels = ex.to_sample(vocab_dict)
            feed_dict = {
                words: x,
                n_words: len(x),
                left: x_left,
                right: x_right,
                labels: x_labels,
                dropout_keep_prob: 1.0
            }
            acc, root_acc = sess.run([accuracy, root_accuracy], feed_dict)
            sum_acc += acc
            sum_root_acc += root_acc
        sum_acc /= test_examples
        sum_root_acc /= test_examples
        print("Accuracy: {:g}, Root accuracy: {:g}".format(sum_acc, sum_root_acc))