#! /usr/bin/env python3
import sys
#sys.path.append('/nfs/home/iperesadin/github/text-classification-tf')

import time
import random

import numpy as np
from tensorflow.contrib import learn

import data_helpers
from tree_based import TreeBased
from flags.train_flags import FLAGS
import pytreebank
from window import lstm_window, cnn_window, dummy_window
from train_helpers import *
from processing.tree_lstm import TreeLstm
from processing.tree_simple import TreeSimple
from processing.subtree_top_k import SubtreeTopK
from outer.subtree_lstm import SubtreeLstm
from outer.outer_composition import OuterComposition
import minibatch
from utils import get_git_revision_hash
from pytreebank import treelstm

print("Current git commit:", get_git_revision_hash().decode('ascii'))

model_parameters = {"dataset_embedding_path",
                    "exclude_leaves_loss",
                    "window_algo",
                    "processing_algo",
                    "recursive_size",
                    "minibatch",
                    "l2_reg_lambda",
                    "consider_only_root"}

print("\nModel Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    if attr in model_parameters:
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
    out.write("Current git commit: " + get_git_revision_hash().decode('ascii') + "\n")
    for attr, value in sorted(FLAGS.__flags.items()):
        out.write("{}={}\n".format(attr.upper(), value))

print("Exclude leaves loss:", FLAGS.exclude_leaves_loss)

# Load data
print("Loading data...")
if FLAGS.task == "SST-1":
    dataset = pytreebank.load_sst(FLAGS.dataset_path)
    x_train = [x.lowercase() for x in dataset["train"]]
    x_dev = [x.lowercase() for x in dataset["dev"]]
    x_test = [x.lowercase() for x in dataset["test"]]
    num_classes = 5
    consider_only_root = False
elif FLAGS.task == "SUBJ":
    with open(os.path.join(FLAGS.dataset_path, 'sents.cparents'), 'r') as pars,\
         open(os.path.join(FLAGS.dataset_path, 'sents.toks'), 'r') as toks:
        x_train = []
        x_dev = []
        x_test = []
        pairs = []
        for x, y in zip(pars, toks):
            p = list(map(int, x.split(" ")))
            t = y.split(" ")
            words = (len(p) + 1) // 2
            assert words == len(t)
            pairs.append((p, t))

    def read_split(file):
        file_path = os.path.join(FLAGS.dataset_path, file + '.split')
        ret = []
        with open(file_path, 'r') as f:
            s = map(int, f.readline().split(" "))
        for i in s:
            ln = len(pairs[i][0])
            lab = [0] * ln
            for j in range(0, ln):
                if pairs[i][0][j] == 0:
                    lab[j] = int(i < 5000)
            ret.append(treelstm.read_tree(pairs[i][0], lab, pairs[i][1]))
        return ret

    x_train = read_split('train')
    x_dev = read_split('dev')
    x_test = read_split('test')

    num_classes = 2
    consider_only_root = True
elif FLAGS.task == "TREC":
    def load_samples(file):
        classes = {'NUM': 0,
                   'LOC': 1,
                   'DESC': 2,
                   'HUM': 3,
                   'ENTY': 4,
                   'ABBR': 5
                   }
        ret = []
        with open(os.path.join(FLAGS.dataset_path, file + '.txt')) as raw,\
             open(os.path.join(FLAGS.dataset_path, 'sents_' + file + '.toks')) as toks,\
             open(os.path.join(FLAGS.dataset_path, 'sents_' + file + '.cparents')) as pars:
            for r, (t, p) in zip(raw, zip(toks, pars)):
                pos = r.find(':')
                lab = classes[r[:pos]]

                p = list(map(int, p.split(" ")))
                t = t.split(" ")
                words = (len(p) + 1) // 2
                assert words == len(t)
                ln = len(p)
                labs = [0] * ln
                for j in range(0, ln):
                    if p[j] == 0: labs[j] = lab

                ret.append(treelstm.read_tree(p, labs, t))
        return ret

    x_train = load_samples('train')
    x_dev = load_samples('test')
    x_test = load_samples('test')

    num_classes=6
    consider_only_root = True
else:
    raise Exception('Unexpected task')

# Build vocabulary
x_train_words = [x.to_words() for x in x_train]
x_train_text = [x.as_text() for x in x_train]

max_document_length = max([len(x) for x in x_train_words])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_train_text)))
vocab_dict = vocab_processor.vocabulary_._mapping
vocab_size = len(vocab_dict)
print("Vocabulary Size: {:d}".format(vocab_size))

# Write vocabulary
vocab_processor.save(os.path.join(out_dir, "vocab"))

dataset_embedding_path = os.path.join(FLAGS.dataset_path, "word2vec.pickle")

word2vec_matrix = None
if FLAGS.embedding_dim is None:  # use pre-trained

    if not os.path.exists(dataset_embedding_path):  # if no word2vec for current dataset
        if "word2vec" in FLAGS.embedding_path:
            print("Loading word2vec...")
            embedding = data_helpers.load_word2vec(FLAGS.embedding_path, vocab_dict)
            print("Loading word2vec finished")
        else:
            print("Loading GLoVe...")
            embedding = data_helpers.load_glove_model(FLAGS.embedding_path, vocab_dict)
            print("Loading GLoVe finished")

        data_helpers.dump_pickle_word_vecs_np(dataset_embedding_path, embedding)
    else:
        print("Loading embedding for dataset...")
        embedding = data_helpers.load_pickle_word_vecs_np(dataset_embedding_path)
        print("Embedding words:", len(embedding))
        print("Embedding dim:", len(list(embedding.items())[0][1]))

    data_helpers.add_unknown_words(embedding, vocab_dict)
    list_vecs = [None] * vocab_size
    for word, idx in vocab_dict.items():
        list_vecs[idx] = embedding[word]
    word2vec_matrix = np.array(list_vecs)

print("Train/Dev/Test split: {:d}/{:d}/{:d}".
      format(len(x_train), len(x_dev), len(x_test)))

def apply_hyperparameters(x_data):
    x_ret = []
    for x in x_data:
        x.set_hyperparameters(num_classes, FLAGS.exclude_leaves_loss)
        s = x.to_sample(vocab_dict)
        x_ret.append(x)
    return x_ret


print("Hyper-parameters setting to trees")
x_train = apply_hyperparameters(x_train)
x_dev = apply_hyperparameters(x_dev)
x_test = apply_hyperparameters(x_test)

n = len(x_train)
print("Hyper-parameters setting to trees finished")

win_sizes_bounds = list(map(int, FLAGS.filter_sizes.split(",")))
if len(win_sizes_bounds) == 1:
    win_sizes_bounds.append(win_sizes_bounds[0])

if len(win_sizes_bounds) != 2:
    print("Wrong win size")
    sys.exit(-1)

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
            win_sizes = [x for x in range(win_sizes_bounds[0], win_sizes_bounds[1] + 1)]
            window_algo = cnn_window.CnnWindow(
                filter_sizes=win_sizes,
                num_filters=FLAGS.num_filters,
                embedding_size=real_embedding_size)
        elif FLAGS.window_algo == "LSTM":
            window_algo = lstm_window.LstmWindow(
                hidden_size=FLAGS.lstm_hidden,
                embedded_size=real_embedding_size,
                min_window_size=win_sizes_bounds[0],
                max_window_size=win_sizes_bounds[1])
        elif FLAGS.window_algo == "DUMMY":
            window_algo = dummy_window.DummyWindow(real_embedding_size)
        else:
            raise Exception('Unknown window algo')

        if FLAGS.processing_algo == "SIMPLE":
            processing_algo = TreeSimple(FLAGS.recursive_size)
        elif FLAGS.processing_algo == "TREE-LSTM":
            processing_algo = TreeLstm(FLAGS.mem_size)
        elif FLAGS.processing_algo == "TOP-K":
            processing_algo = SubtreeTopK(6, backend='LSTM', lstm_hidden=200)
        else:
            raise Exception('Unknown processing algo')
        tree_nn = TreeBased(
            num_classes,
            vocab_size=len(vocab_processor.vocabulary_),
            window_algo=window_algo,
            processing_algo=processing_algo,
            outer_algo=None,
            # outer_algo=OuterComposition([
            #     SubtreeLstm(),
            #     SubtreeTopK(4,
            #                 mode='symbiosis',
            #                 backend='LSTM',
            #                 num_filters=128,
            #                 lstm_hidden=150,
            #                 symbiosis_real_consider=False)
            # ]),
            consider_only_root=consider_only_root,
            exclude_leaves_loss=FLAGS.exclude_leaves_loss,
            embedding_size=FLAGS.embedding_dim,
            pretrained_embedding=word2vec_matrix,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        print("Model is initialized")

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)

        if FLAGS.optimizer == "Adagrad":
            optimizer_ = tf.train.AdagradOptimizer(0.05)
        elif FLAGS.optimizer == "Adam":
            optimizer_ = tf.train.AdamOptimizer(0.001)
        else:
            raise Exception('Unknown optimizer')
        #AdaGrad
        #AdaDelta
        #reg_lam=0.0001

        optimizer = minibatch.MinibatchOptimizer(optimizer_, global_step, tree_nn, num_classes)

        train_summary_writer, dev_summary_writer, test_summary_writer = init_summary_writers(sess, out_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        TRAIN_MEAS_BATCH = 4
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

        batch_size = FLAGS.minibatch
        random.seed(1)
        for epoch in range(FLAGS.num_epochs):
            random.shuffle(x_train)

            for batch_start in range(0, n, batch_size):
                batch = x_train[batch_start:min(n, batch_start + batch_size)]

                lt, at, rlt, rat = train_batch(batch, optimizer, vocab_dict, sess, global_step, FLAGS.dropout_keep_prob)
                sum_train_loss += lt
                sum_train_acc += at
                if rlt:
                    root_cnt += 1
                    sum_train_root_loss += rlt
                    sum_train_root_acc += rat

                current_step = tf.train.global_step(sess, global_step)

                if current_step % TRAIN_MEAS_BATCH == 0:
                    print("{} {}x{}-batch: time {:g} sec, loss {:g}, acc {:g}, root_loss {:g}, root_acc {:g}".
                          format(current_step // TRAIN_MEAS_BATCH,
                                 TRAIN_MEAS_BATCH,
                                 batch_size,
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
                        dev_batch(x_dev, optimizer, vocab_dict, sess, global_step, (dev_summary_writer, current_step // TRAIN_MEAS_BATCH))

                    test_loss, test_acc, test_root_loss, test_root_acc = \
                        dev_batch(x_test, optimizer, vocab_dict, sess, global_step, (test_summary_writer, current_step // TRAIN_MEAS_BATCH))

                    if dev_root_acc > max_dev:
                        max_dev = dev_root_acc
                        max_test = test_root_acc
                        dev_iter = current_step // TRAIN_MEAS_BATCH

                    print("Dev evaluation: loss {:g}, acc {:g}, root_loss {:g}, root_acc {:g}".
                          format(dev_loss, dev_acc, dev_root_loss, dev_root_acc))
                    print("Max dev evaluation root accuracy: {:g} and test {:g}, on batch = {}".
                          format(max_dev, max_test, dev_iter))
                    print("Test evaluation: loss {:g}, acc {:g}, root_loss {:g}, root_acc {:g}".
                          format(test_loss, test_acc, test_root_loss, test_root_acc))
                    print("")

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            print("Epoch #" + str(epoch + 1) + " has finished")
