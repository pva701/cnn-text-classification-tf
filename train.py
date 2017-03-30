#! /usr/bin/env python3

import os
import time
import datetime

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

import data_helpers
from tree_simple import BinaryTreeSimple
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
    for ex in x_dataset:
        l, ac, rl, rac = evaluator(ex)
        sum_loss += l
        sum_acc += ac
        sum_root_loss += rl
        sum_root_acc += rac
        examples += 1
    sum_loss /= examples
    sum_acc /= examples
    sum_root_loss /= examples
    sum_root_acc /= examples
    return sum_loss, sum_acc, sum_root_loss, sum_root_acc

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
dataset = pytreebank.load_sst(FLAGS.sst_path)
x_train = [x.lowercase() for x in dataset["train"]]
x_dev = [x.lowercase() for x in dataset["dev"]]
x_test = [x.lowercase() for x in dataset["test"]]
n = len(x_train)

# Build vocabulary
x_train_words = [x.to_words() for x in x_train]
x_train_text = [x.as_text() for x in x_train]

max_document_length = max([len(x) for x in x_train_words])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_train_text)))
vocab_dict = vocab_processor.vocabulary_._mapping
vocab_size = len(vocab_dict)

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
print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(x_train), len(x_dev), len(x_test)))

print("To sample")
for x in x_train:
    x.to_sample(vocab_dict)

for x in x_dev:
    x.to_sample(vocab_dict)
print("To sample finished")

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        tree_nn = BinaryTreeSimple(
            num_classes=5,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            pretrained_embedding=word2vec_matrix,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        print("Model is initialized")

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(tree_nn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", tree_nn.loss)
        acc_summary = tf.summary.scalar("accuracy", tree_nn.accuracy)
        #root_loss_summary = tf.summary.scalar("root_loss", tree_nn.root_loss)
        #root_acc_summary = tf.summary.scalar("root_accuracy", tree_nn.root_accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_sample(tree):
            """
            A single training step
            """
            x, left, right, labels = tree.to_sample(vocab_dict)
            feed_dict = {
                tree_nn.words: x,
                tree_nn.n_words: len(x),
                tree_nn.left: left,
                tree_nn.right: right,
                tree_nn.labels: labels,
                tree_nn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, root_loss, root_acc = sess.run(
                [train_op,
                 global_step,
                 train_summary_op,
                 tree_nn.loss,
                 tree_nn.accuracy,
                 tree_nn.root_loss,
                 tree_nn.root_accuracy],
                feed_dict)

            train_summary_writer.add_summary(summaries, step)
            return loss, accuracy, root_loss, root_acc

        def dev_sample(tree, writer=None):
            """
            Evaluates model on a dev set
            """
            x, left, right, labels = tree.to_sample(vocab_dict)
            feed_dict = {
                tree_nn.words: x,
                tree_nn.n_words: len(x),
                tree_nn.left: left,
                tree_nn.right: right,
                tree_nn.labels: labels,
                tree_nn.dropout_keep_prob: 1.0
            }

            step, summaries, loss, accuracy, root_loss, root_acc = sess.run(
                [global_step,
                 dev_summary_op,
                 tree_nn.loss,
                 tree_nn.accuracy,
                 tree_nn.root_loss,
                 tree_nn.root_accuracy],
                feed_dict)
            if writer:
                writer.add_summary(summaries, step)
            return loss, accuracy, root_loss, root_acc

        np.random.seed(1)
        TRAIN_MEAS_BATCH = 100
        sum_train_loss = 0.0
        sum_train_root_loss = 0.0
        sum_train_acc = 0.0
        sum_train_root_acc = 0.0
        batch_time = time.time()

        for epoch in range(FLAGS.num_epochs):
            for sample_id in np.random.permutation(n):
                lt, at, rlt, rat = train_sample(x_train[sample_id])
                sum_train_loss += lt
                sum_train_acc += at
                sum_train_root_loss += rlt
                sum_train_root_acc += rat

                current_step = tf.train.global_step(sess, global_step)

                if current_step % TRAIN_MEAS_BATCH == 0:
                    print("{} 100-batch: time {:g} sec, loss {:g}, acc {:g}, root_loss {:g}, root_acc {:g}".
                          format(current_step // TRAIN_MEAS_BATCH,
                                 time.time() - batch_time,
                                 sum_train_loss / TRAIN_MEAS_BATCH,
                                 sum_train_acc / TRAIN_MEAS_BATCH,
                                 sum_train_root_loss / TRAIN_MEAS_BATCH,
                                 sum_train_root_acc / TRAIN_MEAS_BATCH))

                    sum_train_loss = 0.0
                    sum_train_root_loss = 0.0
                    sum_train_acc = 0.0
                    sum_train_root_acc = 0.0
                    batch_time = time.time()

                if current_step % FLAGS.evaluate_every == 0:
                    dev_loss, dev_acc, dev_root_loss, dev_root_acc = \
                        eval_dataset(x_dev, lambda ex: dev_sample(ex, writer=dev_summary_writer))
                    print("Dev evaluation: loss {:g}, acc {:g}, root_loss {:g}, root_acc {:g}".
                          format(dev_loss, dev_acc, dev_root_acc, dev_root_acc))
                    print("")

                if current_step % FLAGS.test_evaluate_every == 0:
                    test_loss, test_acc, test_root_loss, test_root_acc = \
                        eval_dataset(x_test, lambda ex: dev_sample(ex))
                    print("Test evaluation: loss {:g}, acc {:g}, root_loss {:g}, root_acc {:g}".
                          format(test_loss, test_acc, test_root_acc, test_root_acc))
                    print("")

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            print("Epoch #" + str(epoch + 1) + " has finished")