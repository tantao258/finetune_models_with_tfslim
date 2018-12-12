import tensorflow as tf
from nets import vgg
from utils import _load_initial_weights
from tensorflow.contrib.slim import arg_scope


class Vgg19(object):
    def __init__(self, num_classes, train_layers=None, learning_rate=0.001, model="train", weights_path='DEFAULT'):

        """Create the graph of the vgg19 model.
        """

        # Parse input arguments into class variables
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = "./pre_trained_models/vgg_19.ckpt"
        else:
            self.WEIGHTS_PATH = weights_path
        self.train_layers = train_layers

        with tf.variable_scope("input"):
            self.image_size = vgg.vgg_19.default_image_size
            self.x_input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="x_input")
            self.y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        if model == "train" or model == "val":
            with arg_scope(vgg.vgg_arg_scope()):
                self.logits, _ = vgg.vgg_19(self.x_input,
                                            num_classes=num_classes,
                                            is_training=True,
                                            dropout_keep_prob=self.keep_prob
                                            )

        if model == "test":
            with arg_scope(vgg.vgg_arg_scope()):
                self.logits, _ = vgg.vgg_19(self.x_input,
                                            num_classes=num_classes,
                                            is_training=False,
                                            dropout_keep_prob=self.keep_prob
                                            )

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_input))

        with tf.name_scope("train"):

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            var_list = [v for v in tf.trainable_variables() if v.name.split('/')[-2] in train_layers or v.name.split('/')[-3] in train_layers ]
            gradients = tf.gradients(self.loss, var_list)
            self.grads_and_vars = list(zip(gradients, var_list))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars, global_step=self.global_step)

        with tf.name_scope("probability"):
            self.probability = tf.nn.softmax(self.logits, name="probability")

        with tf.name_scope("prediction"):
            self.prediction = tf.argmax(self.logits, 1, name="prediction")

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y_input, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

    def load_initial_weights(self, session):
        _load_initial_weights(session=session,
                              weightPath=self.WEIGHTS_PATH,
                              train_layers=self.train_layers)
