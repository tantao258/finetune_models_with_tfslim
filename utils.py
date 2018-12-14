import os
import cv2
import numpy as np
import tensorflow as tf
from nets import dataset_utils
from tensorflow.python.framework import dtypes
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework.ops import convert_to_tensor
try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib


IMAGENET_MEAN = tf.constant([121.55213, 113.84197, 99.5037], dtype=tf.float32)


class ImageDataGenerator(object):
    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True, buffer_size=1000, img_out_size=224):
        """Create a new ImageDataGenerator.
        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.
        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                  different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data
                in the dataset and the initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        """

        self.txt_file = txt_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # the resize img
        self.img_out_size = img_out_size

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=20)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls=20)

        else:
            raise ValueError("Invalid mode {}" .format(mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)
        data = data.repeat()
        iterator = data.make_one_shot_iterator()
        self.iterator = iterator

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [self.img_out_size, self.img_out_size])
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        # img_bgr = img_centered[:, :, ::-1]

        return img_centered, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [self.img_out_size, self.img_out_size])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        # img_bgr = img_centered[:, :, ::-1]

        return img_centered, one_hot


def _load_initial_weights(session, weightPath, train_layers):
    print("parameters loading ...")

    reader = pywrap_tensorflow.NewCheckpointReader(weightPath)

    # Load the weights into memory
    var_to_shape_map = reader.get_variable_to_shape_map()

    for op_name in var_to_shape_map:
        # Do not load variable: global_step for finetuning
        if op_name == "global_step":
            continue

        op_name_list = op_name.split("/")
        # 判断两个列表是否有交集
        if len([item for item in op_name_list if item in train_layers]) != 0:
            continue

        try:

            with tf.variable_scope("/".join(op_name.split("/")[0:-1]), reuse=True):

                data = reader.get_tensor(op_name)

                var = tf.get_variable(op_name.split("/")[-1], trainable=False)
                session.run(var.assign(data))

        except ValueError:

            tmp1 = list(op_name in str(item) for item in tf.global_variables())
            tmp2 = np.sum([int(item) for item in tmp1])
            if tmp2 == 0:
                print("Don't be loaded: {}, cause: {}".format(op_name, "new model no need this variable."))
            else:
                print("Don't be loaded: {}, cause: {}".format(op_name, ValueError))

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def download_ckpt(url):
    target_dir = os.path.join("./pre_trained_models/")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    dataset_utils.download_and_uncompress_tarball(url, target_dir)


def compute_mean(train_path="./data/train.txt", validation_path="./data/validation.txt"):
    count = 0
    imgs_mean = np.zeros(shape=(3,), dtype=np.float32)
    for filepath in [train_path, validation_path]:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                img = cv2.cvtColor(cv2.imread(line.split(" ")[0]), cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32)
                imgs_mean += np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
                count += 1
    IMAGES_MEAN = imgs_mean / count
    print(IMAGES_MEAN)


if __name__ == "__main__":
    compute_mean()