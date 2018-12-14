import os
import time
import numpy as np
import datetime
import tensorflow as tf
from nets import densenet
from model_densenet_161 import DenseNet_161
from utils import ImageDataGenerator
from utils import download_ckpt

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

"""
Configuration Part.
"""
# Parameters
tf.app.flags.DEFINE_string("train_file", './data/train.txt', "the path of train data")
tf.app.flags.DEFINE_string("val_file", './data/validation.txt', "the path of val data")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learn_rate(default:0.001)")
tf.app.flags.DEFINE_integer("num_epochs", 50, "num_epoches(default:10)")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch_size(default:128)")
tf.app.flags.DEFINE_integer("num_classes", 5, "num_classes(default:2)")
tf.app.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 3, "num_checkpoints(default:3)")
FLAGS = tf.app.flags.FLAGS
num_validation = 10000
train_layers = ["logits"]


"""
Main Part of the finetuning Script.
"""
# Load data on the cpu
print("Loading data...")
with tf.device('/cpu:0'):
    train_iterator = ImageDataGenerator(txt_file=FLAGS.train_file,
                                        mode='training',
                                        batch_size=FLAGS.batch_size,
                                        num_classes=FLAGS.num_classes,
                                        shuffle=True,
                                        img_out_size=densenet.densenet161.default_image_size
                                        )

    val_iterator = ImageDataGenerator(txt_file=FLAGS.val_file,
                                      mode='inference',
                                      batch_size=FLAGS.batch_size,
                                      num_classes=FLAGS.num_classes,
                                      shuffle=False,
                                      img_out_size=densenet.densenet161.default_image_size
                                      )

    train_next_batch = train_iterator.iterator.get_next()
    val_next_batch = val_iterator.iterator.get_next()


# Initialize model
densenet_161 = DenseNet_161(num_classes=FLAGS.num_classes,
                            train_layers=train_layers
                            )

with tf.Session() as sess:
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "densenet_161", timestamp))
    print("Writing to {}\n".format(out_dir))

    # define summary
    grad_summaries = []
    for g, v in densenet_161.grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    loss_summary = tf.summary.scalar("loss", densenet_161.loss)
    acc_summary = tf.summary.scalar("accuracy", densenet_161.accuracy)

    # merge all the train summary
    train_summary_merged = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), graph=sess.graph)
    # merge all the dev summary
    val_summary_merged = tf.summary.merge([loss_summary, acc_summary])
    val_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "val"), graph=sess.graph)

    # checkPoint saver
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "ckpt"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())

    # Load the pre_trained weights into the non-trainable layer
    if "densenet_161.ckpt" not in os.listdir("./pre_trained_models/"):
        print("didn't find the .ckpt pre_trained_model in the directory './pre_trained_models/'")
        exit()

    densenet_161.load_initial_weights(sess)
    print("run the tensorboard in terminal: \ntensorboard --logdir={} --port=6006 \n".format(out_dir))

    while True:
        step = 0
        # train loop
        x_batch_train, y_batch_train = sess.run(train_next_batch)
        _, step, train_summaries, loss, accuracy = sess.run([densenet_161.train_op, densenet_161.global_step, train_summary_merged, densenet_161.loss, densenet_161.accuracy],
                                                            feed_dict={
                                                                densenet_161.x_input: x_batch_train,
                                                                densenet_161.y_input: y_batch_train,
                                                                densenet_161.learning_rate: FLAGS.learning_rate
                                                            })
        train_summary_writer.add_summary(train_summaries, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step: {}, loss: {:g}, acc: {:g}".format(time_str, step, loss, accuracy))

        # validation
        current_step = tf.train.global_step(sess, densenet_161.global_step)

        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            # num_batches in one validation
            num_batchs_one_validation = int(num_validation / FLAGS.batch_size)
            loss_list = []
            acc_list = []

            for i in range(num_batchs_one_validation):

                x_batch_val, y_batch_val = sess.run(val_next_batch)
                step, dev_summaries, loss, accuracy = sess.run([densenet_161.global_step, val_summary_merged, densenet_161.loss_val, densenet_161.accuracy],
                                                               feed_dict={
                                                                   densenet_161.x_input: x_batch_val,
                                                                   densenet_161.y_input: y_batch_val,
                                                               })
                loss_list.append(loss)
                acc_list.append(accuracy)
                val_summary_writer.add_summary(dev_summaries, step)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, acc: {:g}".format(time_str, step, np.mean(loss_list), np.mean(acc_list)))
            print("\n")

        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

        step += 1

        # break conditon
        if current_step == 1600:
            exit()