#! /usr/bin/env python3

import os
import time

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

import data_helpers
from tree_cnn import TreeSimpleCNN
from flags.train_flags import FLAGS
import pytreebank

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def eval_dataset(x_dataset, evaluator):
    sum_loss = 0.0
    sum_root_loss = 0.0
    sum_acc = 0.0
    sum_root_acc = 0.0
    examples = 0
    examples_root = 0

    for ex in x_dataset:
        l, ac, rl, rac = evaluator(ex)
        sum_loss += l
        sum_acc += ac
        if rl:
            examples_root += 1
            sum_root_loss += rl
            sum_root_acc += rac
        examples += 1

    sum_loss /= examples
    sum_acc /= examples
    sum_root_loss /= examples_root
    sum_root_acc /= examples_root
    return sum_loss, sum_acc, sum_root_loss, sum_root_acc

def train_sample(tree, tree_nn, vocab_dict, is_binary, summary=None):
    if is_binary:
        x, left, right, labels, binary_ids = \
            tree.to_sample(vocab_dict, is_binary=True)
        if binary_ids[-1] == 2 * len(x) - 2:
            root_valid = True
        else:
            root_valid = False
    else:
        binary_ids = []
        root_valid = True
        x, left, right, labels = tree.to_sample(vocab_dict)

    feed_dict = {
        tree_nn.words: x,
        tree_nn.n_words: len(x),
        tree_nn.left: left,
        tree_nn.right: right,
        tree_nn.labels: labels,
        tree_nn.binary_ids: binary_ids,
        tree_nn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }

    if summary:
        _, step, summaries, loss, accuracy, root_loss, root_acc = sess.run(
            [train_op, global_step, summary[1], tree_nn.loss, tree_nn.accuracy, tree_nn.root_loss, tree_nn.root_accuracy],
            feed_dict)
    else:
        _, step, loss, accuracy, root_loss, root_acc = sess.run(
            [train_op, global_step, tree_nn.loss, tree_nn.accuracy, tree_nn.root_loss, tree_nn.root_accuracy],
            feed_dict)

    if summary:
        summary[0].add_summary(summaries, step)

    if root_valid:
        return loss, accuracy, root_loss, root_acc
    return loss, accuracy, None, None

def dev_sample(tree, tree_nn, vocab_dict, is_binary, summary=None):
    if is_binary:
        x, left, right, labels, binary_ids = tree.to_sample(vocab_dict, is_binary=True)
        if binary_ids[-1] == 2 * len(x) - 2:
            root_valid = True
        else:
            root_valid = False
    else:
        binary_ids = []
        root_valid = True
        x, left, right, labels = tree.to_sample(vocab_dict)

    feed_dict = {
        tree_nn.words: x,
        tree_nn.n_words: len(x),
        tree_nn.left: left,
        tree_nn.right: right,
        tree_nn.labels: labels,
        tree_nn.binary_ids: binary_ids,
        tree_nn.dropout_keep_prob: 1.0
    }

    if summary:
        step, summaries, loss, accuracy, root_loss, root_acc = sess.run(
            [global_step,
             summary[1],
             tree_nn.loss,
             tree_nn.accuracy,
             tree_nn.root_loss,
             tree_nn.root_accuracy],
            feed_dict)
    else:
        step, loss, accuracy, root_loss, root_acc = sess.run(
            [global_step,
             tree_nn.loss,
             tree_nn.accuracy,
             tree_nn.root_loss,
             tree_nn.root_accuracy],
            feed_dict)

    if summary:
        summary[0].add_summary(summaries, step)

    if root_valid:
        return loss, accuracy, root_loss, root_acc
    return loss, accuracy, None, None

def init_summaries():
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", tree_nn.loss)
    acc_summary = tf.summary.scalar("accuracy", tree_nn.accuracy)
    # root_loss_summary = tf.summary.scalar("root_loss", tree_nn.root_loss)
    # root_acc_summary = tf.summary.scalar("root_accuracy", tree_nn.root_accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
    return train_summary_writer, train_summary_op, dev_summary_writer, dev_summary_op


# Data Preparation
# ==================================================

# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))

# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

is_binary_task = FLAGS.is_binary
print("Binary classification task:", is_binary_task)

# Load data
print("Loading data...")
dataset = pytreebank.load_sst(FLAGS.sst_path)
x_train = [x.lowercase() for x in dataset["train"]]
x_dev = [x.lowercase() for x in dataset["dev"]]
x_test = [x.lowercase() for x in dataset["test"]]

# Build vocabulary
x_train_words = [x.to_words() for x in x_train]
x_train_text = [x.as_text() for x in x_train]

max_document_length = max([len(x) for x in x_train_words])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_train_text)))
vocab_dict = vocab_processor.vocabulary_._mapping
vocab_size = len(vocab_dict)

# Write vocabulary
vocab_processor.save(os.path.join(out_dir, "vocab"))

word2vec_matrix = None
if FLAGS.embedding_dim is None:  # use pretrained
    if not os.path.exists(FLAGS.dataset_word2vec_path):  # if no word2vec for current dataset
        print("Loading word2vec...")
        word2vec = data_helpers.load_word2vec(FLAGS.word2vec_path, vocab_dict)
        print("Loading word2vec finished")

        data_helpers.dump_pickle_word_vecs_np(FLAGS.dataset_word2vec_path, word2vec)
    else:
        print("Loading word2vec for dataset...")
        word2vec = data_helpers.load_pickle_word_vecs_np(FLAGS.dataset_word2vec_path)
        print("Word2Vec words = ", len(word2vec))
        print("Wrod2Vec dim = ", len(list(word2vec.items())[0][1]))

    data_helpers.add_unknown_words(word2vec, vocab_dict)
    list_vecs = [None] * vocab_size
    for word, idx in vocab_dict.items():
        list_vecs[idx] = word2vec[word]
    word2vec_matrix = np.array(list_vecs)
print("Vocabulary Size: {:d}".format(vocab_size))
print("Train/Dev/Test split: {:d}/{:d}/{:d}".
      format(len(x_train), len(x_dev), len(x_test)))

def to_sample_list(x_data):
    x_ret = []
    for x in x_data:
        x.to_sample(vocab_dict, is_binary_task)
        if is_binary_task:
            if len(x.sample_bin[4]) != 0:
                x_ret.append(x)
        else:
            x_ret.append(x)
    return x_ret


print("To sample")
x_train = to_sample_list(x_train)
x_dev = to_sample_list(x_dev)
x_test = to_sample_list(x_test)

n = len(x_train)
print("To sample finished")

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        tree_nn = TreeSimpleCNN(
            is_binary_task,
            vocab_size=len(vocab_processor.vocabulary_),
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            embedding_size=FLAGS.embedding_dim,
            pretrained_embedding=word2vec_matrix,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        print("Model is initialized")

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(tree_nn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        train_summary_writer, train_summary_op, dev_summary_writer, dev_summary_op = init_summaries()

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        np.random.seed(1)
        TRAIN_MEAS_BATCH = 100
        sum_train_loss = 0.0
        sum_train_root_loss = 0.0
        sum_train_acc = 0.0
        sum_train_root_acc = 0.0
        root_cnt = 0
        batch_time = time.time()

        max_dev = 0.0
        max_test = 0.0

        for epoch in range(FLAGS.num_epochs):
            for sample_id in np.random.permutation(n):
                lt, at, rlt, rat = train_sample(x_train[sample_id], tree_nn, vocab_dict,
                                                is_binary_task,
                                                summary=(train_summary_writer, train_summary_op))
                sum_train_loss += lt
                sum_train_acc += at
                if rlt:
                    root_cnt += 1
                    sum_train_root_loss += rlt
                    sum_train_root_acc += rat

                current_step = tf.train.global_step(sess, global_step)

                if current_step % TRAIN_MEAS_BATCH == 0:
                    print("{} 100-batch: time {:g} sec, loss {:g}, acc {:g}, root_loss {:g}, root_acc {:g}".
                          format(current_step // TRAIN_MEAS_BATCH,
                                 time.time() - batch_time,
                                 sum_train_loss / TRAIN_MEAS_BATCH,
                                 sum_train_acc / TRAIN_MEAS_BATCH,
                                 sum_train_root_loss / root_cnt,
                                 sum_train_root_acc / root_cnt))
                    root_cnt = 0
                    sum_train_loss = 0.0
                    sum_train_root_loss = 0.0
                    sum_train_acc = 0.0
                    sum_train_root_acc = 0.0
                    batch_time = time.time()

                if current_step % FLAGS.evaluate_every == 0:
                    dev_loss, dev_acc, dev_root_loss, dev_root_acc = \
                        eval_dataset(x_dev, lambda ex: dev_sample(ex, tree_nn, vocab_dict, is_binary_task,
                                                                  summary=(dev_summary_writer, dev_summary_op)))
                    print("Dev evaluation: loss {:g}, acc {:g}, root_loss {:g}, root_acc {:g}".
                          format(dev_loss, dev_acc, dev_root_acc, dev_root_acc))
                    max_dev = max(max_dev, dev_root_acc)
                    print("Max dev evaluation root accuracy: {:g}".format(max_dev))
                    print("")

                if current_step % FLAGS.test_evaluate_every == 0:
                    test_loss, test_acc, test_root_loss, test_root_acc = \
                        eval_dataset(x_test, lambda ex: dev_sample(ex, tree_nn, vocab_dict, is_binary_task))
                    print("Test evaluation: loss {:g}, acc {:g}, root_loss {:g}, root_acc {:g}".
                          format(test_loss, test_acc, test_root_acc, test_root_acc))
                    max_test = max(max_test, test_root_acc)
                    print("Max test evaluation root accuracy: {:g}".format(max_test))
                    print("")

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            print("Epoch #" + str(epoch + 1) + " has finished")
