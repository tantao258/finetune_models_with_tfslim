import os
import numpy as np
import tensorflow as tf
from nets import inception
from model_inceptionv1 import InceptionV1
from utils import ImageDataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

"""
Configuration Part.
"""
# Parameters
tf.app.flags.DEFINE_string("test_file", './data/test.txt', "the path of test data")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch_size(default:128)")
tf.app.flags.DEFINE_integer("num_classes", 5, "num_classes(default:2)")
FLAGS = tf.app.flags.FLAGS
num_validation = 10000
train_layers = ["Conv2d_0c_1x1"]

# Load data on the cpu
print("Loading data...")
with tf.device('/cpu:0'):
    test_iterator = ImageDataGenerator(txt_file=FLAGS.test_file,
                                       mode='inference',
                                       batch_size=FLAGS.batch_size,
                                       num_classes=FLAGS.num_classes,
                                       shuffle=True,
                                       img_out_size=inception.inception_v1.default_image_size
                                       )
    test_next_batch = test_iterator.iterator.get_next()


# Initialize model
inceptionv1 = InceptionV1(num_classes=FLAGS.num_classes)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=tf.global_variables())
    model_file = tf.train.latest_checkpoint("./runs/inceptionv1/1544497939/ckpt/")
    saver.restore(sess, model_file)

    num_batchs_one_validation = int(num_validation / FLAGS.batch_size)
    acc_list = []
    for i in range(num_batchs_one_validation):
        x_batch_test, y_batch_test = sess.run(test_next_batch)
        accuracy = sess.run(inceptionv1.accuracy, feed_dict={inceptionv1.x_input: x_batch_test,
                                                             inceptionv1.y_input: y_batch_test,
                                                             inceptionv1.keep_prob: 1.0
                                                             }
                            )
        acc_list.append(accuracy)
    print("accuracy on test dataSet: {}".format(np.mean(acc_list)))

