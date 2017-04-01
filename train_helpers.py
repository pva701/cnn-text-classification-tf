__author__ = 'pva701'

import tensorflow as tf
import os


def train_sample(tree, tree_nn, vocab_dict, is_binary, sess, train_op, global_step, dropout_k):
    x, left, right, labels, weights, binary_ids = tree.to_sample(vocab_dict)

    if is_binary:
        if binary_ids[-1] == 2 * len(x) - 2:
            root_valid = True
        else:
            root_valid = False
    else:
        binary_ids = []
        root_valid = True

    feed_dict = {
        tree_nn.words: x,
        tree_nn.n_words: len(x),
        tree_nn.left: left,
        tree_nn.right: right,
        tree_nn.labels: labels,
        tree_nn.binary_ids: binary_ids,
        tree_nn.weights_loss: weights,
        tree_nn.dropout_keep_prob: dropout_k
    }

    _, _, loss, accuracy, root_loss, root_acc = sess.run(
        [train_op, global_step, tree_nn.loss, tree_nn.accuracy,
         tree_nn.root_loss, tree_nn.root_accuracy],
        feed_dict)

    if root_valid:
        return loss, accuracy, root_loss, root_acc
    return loss, accuracy, None, None

def dev_sample(tree, tree_nn, vocab_dict, is_binary, sess, global_step):
    x, left, right, labels, weights, binary_ids = tree.to_sample(vocab_dict)

    if is_binary:
        if binary_ids[-1] == 2 * len(x) - 2:
            root_valid = True
        else:
            root_valid = False
    else:
        binary_ids = []
        root_valid = True

    feed_dict = {
        tree_nn.words: x,
        tree_nn.n_words: len(x),
        tree_nn.left: left,
        tree_nn.right: right,
        tree_nn.labels: labels,
        tree_nn.binary_ids: binary_ids,
        tree_nn.weights_loss: weights,
        tree_nn.dropout_keep_prob: 1.0
    }

    _, loss, accuracy, root_loss, root_acc = sess.run(
        [global_step, tree_nn.loss, tree_nn.accuracy,
         tree_nn.root_loss, tree_nn.root_accuracy],
        feed_dict)

    if root_valid:
        return loss, accuracy, root_loss, root_acc
    return loss, accuracy, None, None

def init_summary_writers(sess, out_dir):
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph, flush_secs=60)

    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph, flush_secs=60)

    test_summary_dir = os.path.join(out_dir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph, flush_secs=60)
    return train_summary_writer, dev_summary_writer, test_summary_writer

def eval_dataset(x_dataset, evaluator, summary=None):
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

    if summary:
        add_summary(sum_loss, sum_acc, sum_root_loss, sum_root_acc, summary)

    return sum_loss, sum_acc, sum_root_loss, sum_root_acc

def add_summary(sum_loss, sum_acc, sum_root_loss, sum_root_acc, summary):
    writer = summary[0]
    step = summary[1]
    summary_loss = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=sum_loss), ])
    summary_acc = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=sum_acc), ])
    summary_root_loss = tf.Summary(value=[tf.Summary.Value(tag="root_loss", simple_value=sum_root_loss), ])
    summary_root_acc = tf.Summary(value=[tf.Summary.Value(tag="root_accuracy", simple_value=sum_root_acc), ])
    writer.add_summary(summary_loss, step)
    writer.add_summary(summary_acc, step)
    writer.add_summary(summary_root_loss, step)
    writer.add_summary(summary_root_acc, step)