#! /usr/bin/env python3

import time

import numpy as np
from tensorflow.contrib import learn

import data_helpers
from tree_based import TreeBased
from flags.train_flags import FLAGS
import pytreebank
import cnn_window
import lstm_window
from train_helpers import *

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


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

descr_file = os.path.join(out_dir, 'descr-' + str(timestamp) + '.txt')
with open(descr_file, 'w+') as out:
    for attr, value in sorted(FLAGS.__flags.items()):
        out.write("{}={}\n".format(attr.upper(), value))

is_binary_task = FLAGS.is_binary
print("Binary classification task:", is_binary_task)
print("Exclude leaves loss:", FLAGS.exclude_leaves_loss)

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
        print("Word2Vec words:", len(word2vec))
        print("Wrod2Vec dim:", len(list(word2vec.items())[0][1]))

    data_helpers.add_unknown_words(word2vec, vocab_dict)
    list_vecs = [None] * vocab_size
    for word, idx in vocab_dict.items():
        list_vecs[idx] = word2vec[word]
    word2vec_matrix = np.array(list_vecs)
print("Vocabulary Size: {:d}".format(vocab_size))
print("Train/Dev/Test split: {:d}/{:d}/{:d}".
      format(len(x_train), len(x_dev), len(x_test)))

def apply_hyperparameters(x_data):
    x_ret = []
    for x in x_data:
        x.set_hyperparameters(is_binary_task, FLAGS.exclude_leaves_loss, FLAGS.weight_loss)
        res = x.to_sample(vocab_dict)

        if is_binary_task:
            if len(res[4]) != 0:
                x_ret.append(x)
        else:
            x_ret.append(x)
    return x_ret


print("Hyper-parameters setting to trees")
x_train = apply_hyperparameters(x_train)
x_dev = apply_hyperparameters(x_dev)
x_test = apply_hyperparameters(x_test)

n = len(x_train)
print("Hyper-parameters setting to trees finished")

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        real_embedding_size = FLAGS.embedding_dim if FLAGS.embedding_dim else word2vec_matrix.shape[1]

        if FLAGS.window_algo == "CNN":
            window_algo = cnn_window.CnnWindow(
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                embedding_size=real_embedding_size)
        elif FLAGS.window_algo == "LSTM":
            window_algo = lstm_window.LstmWindow(
                hidden_size=FLAGS.lstm_hidden,
                embedded_size=real_embedding_size)
        else:
            raise Exception('Unknown window algo')

        tree_nn = TreeBased(
            is_binary_task,
            vocab_size=len(vocab_processor.vocabulary_),
            recursive_size=FLAGS.recursive_size,
            window_algo=window_algo,
            exclude_leaves_loss=FLAGS.exclude_leaves_loss,
            is_weight_loss=FLAGS.weight_loss,
            embedding_size=FLAGS.embedding_dim,
            pretrained_embedding=word2vec_matrix,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        print("Model is initialized")

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)

        if FLAGS.optimizer == "Adagrad":
            optimizer = tf.train.AdagradOptimizer(0.05)
        elif FLAGS.optimizer == "Adam":
            optimizer = tf.train.AdamOptimizer(0.001)
        else:
            raise Exception('Unknown optimizer')
        #AdaGrad
        #AdaDelta
        #reg_lam=0.0001

        grads_and_vars = optimizer.compute_gradients(tree_nn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        train_summary_writer, dev_summary_writer, test_summary_writer = init_summary_writers(sess, out_dir)

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
        dev_iter = 0
        test_iter = 0

        for epoch in range(FLAGS.num_epochs):
            for sample_id in np.random.permutation(n):
                lt, at, rlt, rat = train_sample(x_train[sample_id], tree_nn, vocab_dict, is_binary_task,
                                                sess, train_op, global_step, FLAGS.dropout_keep_prob)
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

                    add_summary(sum_train_loss / TRAIN_MEAS_BATCH,
                                sum_train_acc / TRAIN_MEAS_BATCH,
                                sum_train_root_loss / root_cnt,
                                sum_train_root_acc / root_cnt,
                                (train_summary_writer, current_step // TRAIN_MEAS_BATCH))

                    root_cnt = 0
                    sum_train_loss = 0.0
                    sum_train_root_loss = 0.0
                    sum_train_acc = 0.0
                    sum_train_root_acc = 0.0
                    batch_time = time.time()

                if current_step % FLAGS.evaluate_every == 0:
                    dev_loss, dev_acc, dev_root_loss, dev_root_acc = \
                        eval_dataset(x_dev, lambda ex: dev_sample(ex, tree_nn, vocab_dict, is_binary_task,
                                                                  sess, global_step),
                                     (dev_summary_writer, current_step // TRAIN_MEAS_BATCH))
                    print("Dev evaluation: loss {:g}, acc {:g}, root_loss {:g}, root_acc {:g}".
                          format(dev_loss, dev_acc, dev_root_loss, dev_root_acc))
                    if dev_root_acc > max_dev:
                        max_dev = dev_root_acc
                        dev_iter = current_step // TRAIN_MEAS_BATCH

                    print("Max dev evaluation root accuracy: {:g}, on batch = {}".format(max_dev, dev_iter))
                    print("")

                if current_step % FLAGS.test_evaluate_every == 0:
                    test_loss, test_acc, test_root_loss, test_root_acc = \
                        eval_dataset(x_test, lambda ex: dev_sample(ex, tree_nn, vocab_dict, is_binary_task,
                                                                   sess, global_step),
                                     (test_summary_writer, current_step // TRAIN_MEAS_BATCH))
                    print("Test evaluation: loss {:g}, acc {:g}, root_loss {:g}, root_acc {:g}".
                          format(test_loss, test_acc, test_root_loss, test_root_acc))
                    if test_root_acc > max_test:
                        max_test = test_root_acc
                        test_iter = current_step // TRAIN_MEAS_BATCH

                    print("Max test evaluation root accuracy: {:g}, on batch = {}".format(max_test, test_iter))
                    print("")

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            print("Epoch #" + str(epoch + 1) + " has finished")
