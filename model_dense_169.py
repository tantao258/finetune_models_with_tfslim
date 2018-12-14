import tensorflow as tf
from nets import densenet
from utils import _load_initial_weights
from utils import average_gradients


class DenseNet_169(object):
    def __init__(self, num_classes, batch_size, train_layers=None, weights_path='DEFAULT'):

        """Create the graph of the densenet_169 model.
        """

        # Parse input arguments into class variables
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = "./pre_trained_models/densenet_169.ckpt"
        else:
            self.WEIGHTS_PATH = weights_path
        self.train_layers = train_layers

        with tf.device("/cpu:0"):
            with tf.variable_scope("input"):
                self.image_size = densenet.densenet_169.default_image_size
                self.x_input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="x_input")
                self.y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")
                self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
                self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        with tf.variable_scope("train"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            tower_grads = []
            loss_total = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(4):
                    with tf.device("/gpu:%d" % i):
                        with tf.name_scope("tower_%d" % i):
                            _x = self.x_input[i * batch_size: (i + 1) * batch_size]
                            _y = self.y_input[i * batch_size:(i + 1) * batch_size]

                            logits = densenet.densenet_169(_x, num_classes=num_classes, is_training=True,
                                                           dropout_keep_prob=self.keep_prob)

                            tf.get_variable_scope().reuse_variables()
                            with tf.name_scope("loss_%d" % i):
                                loss = tf.reduce_mean(
                                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=_y))
                                loss_total.append(loss)
                            with tf.name_scope("grade_%d" % i):
                                grads = optimizer.compute_gradients(loss)
                                tower_grads.append(grads)

            self.grads_and_vars = average_gradients(tower_grads)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars,
                                                          global_step=self.global_step)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(loss_total)

        with tf.device("/gpu:3"):

            logits_val = densenet.densenet_169(self.x_input, num_classes=num_classes, is_training=False,
                                               dropout_keep_prob=1.0)

            with tf.name_scope("probability"):
                self.probability = tf.nn.softmax(logits_val, name="probability")

            with tf.name_scope("prediction"):
                self.prediction = tf.argmax(logits_val, 1, name="prediction")

            with tf.name_scope("accuracy"):
                correct_prediction = tf.equal(self.prediction, tf.argmax(self.y_input, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

    def load_initial_weights(self, session):
        _load_initial_weights(session=session,
                              weightPath=self.WEIGHTS_PATH,
                              train_layers=self.train_layers)