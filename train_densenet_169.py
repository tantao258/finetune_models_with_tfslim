import os
import time
import datetime
import tensorflow as tf
from nets import densenet
from model_dense_169 import DenseNet_169
from utils import ImageDataGenerator
from utils import download_ckpt

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


"""
Configuration Part.
"""
# Parameters
tf.app.flags.DEFINE_string("train_file", './cifar_data/train.txt', "the path of train data")
tf.app.flags.DEFINE_string("val_file", './cifar_data/validation.txt', "the path of val data")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learn_rate(default:0.001)")
tf.app.flags.DEFINE_integer("num_epochs", 50, "num_epoches(default:10)")
tf.app.flags.DEFINE_integer("batch_size", 3, "batch_size(default:128)")
tf.app.flags.DEFINE_integer("num_classes", 10, "num_classes(default:2)")
tf.app.flags.DEFINE_float("keep_prob", 0.8, "dropout_rate(default:0.8)")
tf.app.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 3, "num_checkpoints(default:3)")
FLAGS = tf.app.flags.FLAGS
# training model with no pre_model, train_layer=[]
train_layers = []

"""
Main Part of the training Script.
"""

# Load data on the cpu
print("Loading data...")
with tf.device('/cpu:0'):
    train_iterator = ImageDataGenerator(txt_file=FLAGS.train_file,
                                        mode='training',
                                        batch_size=FLAGS.batch_size,
                                        num_classes=FLAGS.num_classes,
                                        shuffle=True,
                                        img_out_size=densenet.densenet_169.default_image_size
                                        )

    val_iterator = ImageDataGenerator(txt_file=FLAGS.val_file,
                                      mode='inference',
                                      batch_size=FLAGS.batch_size,
                                      num_classes=FLAGS.num_classes,
                                      shuffle=False,
                                      img_out_size=densenet.densenet_169.default_image_size
                                      )

    train_next_batch = train_iterator.iterator.get_next()
    val_next_batch = val_iterator.iterator.get_next()


# Initialize model
densenet_169 = DenseNet_169(num_classes=FLAGS.num_classes,
                            train_layers=train_layers,
                            model="train"
                            )

with tf.Session() as sess:
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "densenet_169", timestamp))
    print("Writing to {}\n".format(out_dir))

    # define summary
    # grad_summaries = []
    # for g, v in densenet_169.grads_and_vars:
    #     if g is not None:
    #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
    #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
    #         grad_summaries.append(grad_hist_summary)
    #         grad_summaries.append(sparsity_summary)
    # grad_summaries_merged = tf.summary.merge(grad_summaries)
    loss_summary = tf.summary.scalar("loss", densenet_169.loss)
    acc_summary = tf.summary.scalar("accuracy", densenet_169.accuracy)

    # merge all the train summary
    train_summary_merged = tf.summary.merge([loss_summary, acc_summary])
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

    while True:
        # train loop
        x_batch_train, y_batch_train = sess.run(train_next_batch)
        _, step, train_summaries, loss, accuracy = sess.run([densenet_169.train_op, densenet_169.global_step, train_summary_merged, densenet_169.loss, densenet_169.accuracy],
                                                            feed_dict={
                                                                densenet_169.x_input: x_batch_train,
                                                                densenet_169.y_input: y_batch_train,
                                                                densenet_169.keep_prob: FLAGS.keep_prob,
                                                                densenet_169.learning_rate: FLAGS.learning_rate
                                                            })
        train_summary_writer.add_summary(train_summaries, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step: {}, loss: {:g}, acc: {:g}".format(time_str, step, loss, accuracy))

        # validation
        current_step = tf.train.global_step(sess, densenet_169.global_step)

        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            x_batch_val, y_batch_val = sess.run(val_next_batch)

            step, dev_summaries, loss, accuracy = sess.run([densenet_169.global_step, val_summary_merged, densenet_169.loss, densenet_169.accuracy],
                                                           feed_dict={
                                                                densenet_169.x_input: x_batch_val,
                                                                densenet_169.y_input: y_batch_val,
                                                                densenet_169.keep_prob: 1
                                                           })
            val_summary_writer.add_summary(dev_summaries, step)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, acc: {:g}".format(time_str, step, loss, accuracy))
            print("\n")

        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))