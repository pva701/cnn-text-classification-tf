__author__ = 'pva701'

import tensorflow as tf
import os
from flags.train_flags import FLAGS
import collections

def create_batch_placeholder(batch, vocab_dict):
    words = []
    n_words = []
    left = []
    right = []
    labels = []
    l_bound = []
    r_bound = []
    euler = []
    euler_l = []
    euler_r = []

    max_len = 0
    max_lab = 0
    for tree in batch:
        x = tree.to_sample(vocab_dict)
        max_len = max(max_len, len(x.words))
        max_lab = max(max_lab, len(x.labels))

        n_words.append(len(x.words))
        words.append(x.words)
        left.append(x.left)
        right.append(x.right)
        l_bound.append(x.l_bound), r_bound.append(x.r_bound)
        labels.append(x.labels)

        euler.append(x.euler)
        euler_l.append(x.euler_l)
        euler_r.append(x.euler_r)

    def populate(c, v, val=-1):
        ret = [e for e in v]
        while len(ret) < c:
            ret.append(val)
        return ret

    words = [populate(max_len, e) for e in words]
    left = [populate(max_len - 1, e) for e in left]
    right = [populate(max_len - 1, e) for e in right]
    l_bound = [populate(max_len - 1, e) for e in l_bound]
    r_bound = [populate(max_len - 1, e) for e in r_bound]
    labels = [populate(max_lab, e, labels[0][0]) for e in labels]
    euler = [populate(2 * max_len - 1, e) for e in euler]
    euler_l = [populate(max_len - 1, e) for e in euler_l]
    euler_r = [populate(max_len - 1, e) for e in euler_r]
    return n_words, words, left, right, l_bound, r_bound, labels, euler, euler_l, euler_r

def train_batch(batch, optimizer, vocab_dict, sess, global_step, dropout_k):
    root_valid = True
    n_words, words, left, right, l_bound, r_bound, labels, euler, euler_l, euler_r = \
        create_batch_placeholder(batch, vocab_dict)
    feed_dict = {
        optimizer.batch_size: len(batch),
        optimizer.words: words,
        optimizer.n_words: n_words,
        optimizer.left: left,
        optimizer.right: right,
        optimizer.l_bound: l_bound,
        optimizer.r_bound: r_bound,
        optimizer.labels: labels,
        optimizer.euler: euler,
        optimizer.euler_l: euler_l,
        optimizer.euler_r: euler_r,
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


def dev_batch(batch, optimizer, vocab_dict, sess, global_step, summary):
    if not FLAGS.with_sent_stat:
        n_words, words, left, right, l_bound, r_bound, labels, euler, euler_l, euler_r = \
            create_batch_placeholder(batch, vocab_dict)

        feed_dict = {
            optimizer.batch_size: len(batch),
            optimizer.words: words,
            optimizer.n_words: n_words,
            optimizer.left: left,
            optimizer.right: right,
            optimizer.l_bound: l_bound,
            optimizer.r_bound: r_bound,
            optimizer.labels: labels,
            optimizer.euler: euler,
            optimizer.euler_l: euler_l,
            optimizer.euler_r: euler_r,
            optimizer.dropout_keep_prob: 1.0
        }

        _, result = sess.run(
            [global_step, optimizer.result],
            feed_dict)
    else:
        test_examples = 0
        sum_acc = 0.0
        result = (0.0, 0.0, 0.0, 0.0)
        sent_statistic = collections.defaultdict(lambda: 0)
        sent_total = collections.defaultdict(lambda: 0)
        g_step = 0
        for x in batch:
            s = x.to_sample(vocab_dict)
            feed_dict = {
                optimizer.batch_size: 1,
                optimizer.words: [s.words],
                optimizer.n_words: [len(s.words)],
                optimizer.left: [s.left],
                optimizer.right: [s.right],
                optimizer.l_bound: [s.l_bound],
                optimizer.r_bound: [s.r_bound],
                optimizer.labels: [s.labels],
                optimizer.euler: [s.euler],
                optimizer.euler_l: [s.euler_l],
                optimizer.euler_r: [s.euler_r],
                optimizer.dropout_keep_prob: 1.0
            }
            g_step, loc_result = sess.run([global_step, optimizer.result], feed_dict)
            result = tuple(x + y for x, y in zip(result, loc_result))
            sum_acc += loc_result[3]
            test_examples += 1
            l = len(s.words)
            sent_total[l] += 1
            sent_statistic[l] += loc_result[3]
        result = tuple(x/test_examples for x in result)
        with open(os.path.join(FLAGS.outdir, 'sent_stat-{}').format(g_step), 'w') as out:
            max_len = max(sent_total.values())
            for l in range(1, max_len + 1):
                if l in sent_total:
                    out.write("{} {} {:g}\n".format(l, sent_total[l], sent_statistic[l] / sent_total[l]))
                else:
                    out.write("{} 0 0\n".format(l))

    loss = result[0]
    root_loss = result[1]
    accuracy = result[2]
    root_acc = result[3]

    if summary:
        add_summary(loss, accuracy, root_loss, root_acc, summary)

    return loss, accuracy, root_loss, root_acc


def init_summary_writers(sess, out_dir):
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph, flush_secs=60)

    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph, flush_secs=60)

    test_summary_dir = os.path.join(out_dir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph, flush_secs=60)
    return train_summary_writer, dev_summary_writer, test_summary_writer


def add_summary(sum_loss, sum_acc, sum_root_loss, sum_root_acc, summary):
    writer = summary[0]
    step = summary[1]
    summary_loss = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=sum_loss.astype(float)), ])
    summary_acc = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=sum_acc.astype(float)), ])
    summary_root_loss = tf.Summary(
        value=[tf.Summary.Value(tag="root_loss", simple_value=sum_root_loss.astype(float)), ])
    summary_root_acc = tf.Summary(
        value=[tf.Summary.Value(tag="root_accuracy", simple_value=sum_root_acc.astype(float)), ])
    writer.add_summary(summary_loss, step)
    writer.add_summary(summary_acc, step)
    writer.add_summary(summary_root_loss, step)
    writer.add_summary(summary_root_acc, step)
