#! /usr/bin/env python

import os

import tensorflow as tf
from tensorflow.contrib import learn
from multiprocessing import Process
from dataset_loaders import load_trec

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("task", "TREC", "Task")
tf.flags.DEFINE_string("dataset_path", "./data/trec", "Dataset path")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1495536797/checkpoints/",
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
if FLAGS.task == "TREC":
    _, _, x_test = load_trec(FLAGS.dataset_path)
    num_classes = 6
else:
    raise Exception('Unknown task')

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
vocab_dict = vocab_processor.vocabulary_._mapping
vocab_size = len(vocab_dict)
print("Vocab size: {}".format(vocab_size))

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)


def restore_and_eval():
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            operation_timeout_in_ms=60000)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            batch_size = graph.get_operation_by_name("batch_size").outputs[0]
            words = graph.get_operation_by_name("words").outputs[0]
            n_words = graph.get_operation_by_name("n_words").outputs[0]
            left = graph.get_operation_by_name("left").outputs[0]
            right = graph.get_operation_by_name("right").outputs[0]
            l_bound = graph.get_operation_by_name("l_bound").outputs[0]
            r_bound = graph.get_operation_by_name("r_bound").outputs[0]
            labels = graph.get_operation_by_name("labels").outputs[0]
            euler = graph.get_operation_by_name("Placeholder").outputs[0]
            euler_l = graph.get_operation_by_name("Placeholder_1").outputs[0]
            euler_r = graph.get_operation_by_name("Placeholder_2").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # for x in graph.get_operations():
            # print(x.name)
            # Tensors we want to evaluate
            # predictions = graph.get_operation_by_name("loss/predictions").outputs[0]
            # accuracy = graph.get_operation_by_name("internal-state_1/accuracy/accuracy").outputs[0]
            root_accuracy = graph.get_operation_by_name("map/while/internal-state/accuracy/root_accuracy").outputs[0]
            print("Got all operations")

            test_examples = 0
            sum_acc = 0.0
            for x in x_test:
                x.set_hyperparameters(num_classes, False)

                s = x.to_sample(vocab_dict)
                feed_dict = {
                    batch_size: 1,
                    words: [s.words],
                    n_words: [len(s.words)],
                    left: [s.left],
                    right: [s.right],
                    l_bound: [s.l_bound],
                    r_bound: [s.r_bound],
                    labels: [s.labels],
                    euler: [s.euler],
                    euler_l: [s.euler_l],
                    euler_r: [s.euler_r],
                    dropout_keep_prob: 1.0
                }
                print("Running session")
                root_acc = sess.run([root_accuracy], feed_dict)
                print("Finish running")
                sum_acc += root_acc
                test_examples += 1
                print(test_examples)

                if test_examples % 100 == 0:
                    print("{} Done".format(test_examples))

            sum_acc /= test_examples
            print("Root Accuracy: {:g}".format(sum_acc))


# def subprocess():
#     restore_and_eval()
#
#
# p = Process(target=subprocess)
# p.daemon = True
# p.start()
# p.join()

restore_and_eval()