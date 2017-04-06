__author__ = 'pva701'

import tensorflow as tf
import os


def create_batch_placeholder(batch, vocab_dict):
    words = []
    n_words = []
    left = []
    right = []
    labels = []
    weights = []
    max_len = 0
    max_lab = 0
    for tree in batch:
        x, left_x, right_x, labels_x, weights_x, _ = tree.to_sample(vocab_dict)
        max_len = max(max_len, len(x))
        max_lab = max(max_lab, len(labels_x))

        n_words.append(len(x))
        words.append(x)
        left.append(left_x)
        right.append(right_x)
        labels.append(labels_x)
        weights.append(weights_x)

    def populate(c, v, val=-1):
        ret = [e for e in v]
        while len(ret) < c:
            ret.append(val)
        return ret

    words = [populate(max_len, e) for e in words]
    left = [populate(max_len, e) for e in left]
    right = [populate(max_len, e) for e in right]
    labels = [populate(max_lab, e, labels[0][0]) for e in labels]
    weights = [populate(max_lab, e) for e in weights]
    return n_words, words, left, right, labels, weights


def train_batch(batch, optimizer, vocab_dict, is_binary, sess, global_step, dropout_k):
    # if is_binary:
    #     if binary_ids[-1] == 2 * len(x) - 2:
    #         root_valid = True
    #     else:
    #         root_valid = False
    # else:
    #     binary_ids = []
    #     root_valid = True

    root_valid = True
    n_words, words, left, right, labels, weights = create_batch_placeholder(batch, vocab_dict)
    feed_dict = {
        optimizer.batch_size: len(batch),
        optimizer.words: words,
        optimizer.n_words: n_words,
        optimizer.left: left,
        optimizer.right: right,
        optimizer.labels: labels,
        # optimizer.binary_ids: binary_ids,
        optimizer.weights_loss: weights,
        optimizer.dropout_keep_prob: dropout_k
    }
    _, _, result = sess.run(
        [optimizer.train_op, global_step, optimizer.result],
        feed_dict)

    loss = result[0]
    root_loss = result[1]
    accuracy = result[2]
    root_acc = result[3]

    if root_valid:
        return loss, accuracy, root_loss, root_acc
    return loss, accuracy, None, None


def dev_batch(batch, optimizer, vocab_dict, is_binary, sess, global_step, summary):
    # if is_binary:
    #     if binary_ids[-1] == 2 * len(x) - 2:
    #         root_valid = True
    #     else:
    #         root_valid = False
    # else:
    #     binary_ids = []
    #     root_valid = True

    root_valid = True
    n_words, words, left, right, labels, weights = create_batch_placeholder(batch, vocab_dict)

    feed_dict = {
        optimizer.batch_size: len(batch),
        optimizer.words: words,
        optimizer.n_words: n_words,
        optimizer.left: left,
        optimizer.right: right,
        optimizer.labels: labels,
        # optimizer.binary_ids: binary_ids,
        optimizer.weights_loss: weights,
        optimizer.dropout_keep_prob: 1.0
    }

    _, result = sess.run(
        [global_step, optimizer.result],
        feed_dict)

    loss = result[0]
    root_loss = result[1]
    accuracy = result[2]
    root_acc = result[3]

    if summary:
        add_summary(loss, accuracy, root_loss, root_acc, summary)

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
