# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for densenet classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
from tensorflow.contrib.slim import max_pool2d


def densenet_base(inputs,
                  growth_rate_k=32,
                  block_list=[6, 12, 32, 32],
                  bc_mode=False,
                  reduction=1.0,
                  ):

    with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):

        net = slim.conv2d(inputs, growth_rate_k * 2, [7, 7], stride=2, scope="con_1", reuse=tf.AUTO_REUSE)

        net = slim.max_pool2d(net, [3, 3], stride=2, scope="pool_1")

        for i, layer_num in enumerate(block_list, start=1):

            with tf.variable_scope("Block_%d" % i, reuse=tf.AUTO_REUSE):
                net = add_block(net, growth_rate_k, layer_num, bc_mode)

            # last block exist without transition layer
            if i != len(block_list) - 1:
                with tf.variable_scope("Transition_%d" % i, reuse=tf.AUTO_REUSE):
                    net = transition_layer(net, reduction)

        return net


def add_block(_input, growth_rate_k, num_layer, bc_mode):
    output = _input
    for layer in range(num_layer):
        with tf.variable_scope("layer_%d" % (layer+1)):
            output = add_internal_layer(output, growth_rate_k, bc_mode=bc_mode)
    return output


def add_internal_layer(_input, growth_rate_k, bc_mode):

    # 3 x 3
    if not bc_mode:
        comp_out = composite_function(_input, out_features=growth_rate_k, kernel_size=3)

    # 1 x 1
    # 3 x 3
    elif bc_mode:
        bottleneck_out = bottleneck(_input, out_features=growth_rate_k * 4)
        comp_out = composite_function(bottleneck_out, out_features=growth_rate_k, kernel_size=3)

    output = tf.concat(values=[_input, comp_out], axis=3)
    return output


def conv(_input, out_features, kernel_size, stride=1):
    output = slim.batch_norm(_input)
    output = tf.nn.relu(output)
    return slim.conv2d(inputs=output, num_outputs=out_features, kernel_size=kernel_size, stride=stride)


def composite_function(_input, out_features, kernel_size=3):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    - dropout, if required
    """
    with tf.variable_scope("composite_function"):
        output = conv(_input=_input, out_features=out_features, kernel_size=kernel_size)
        output = slim.dropout(inputs=output)
    return output


def bottleneck(_input, out_features):
    with tf.variable_scope("bottleneck"):
        output = conv(_input=_input, out_features=out_features, kernel_size=1, stride=1)
        output = slim.dropout(inputs=output)
    return output


def transition_layer(_input, reduction):
    """Call H_l composite function with 1x1 kernel and after average
    pooling
    """
    # call composite function with 1x1 kernel
    out_features = int(int(_input.get_shape()[-1]) * reduction)
    output = composite_function(_input, out_features=out_features, kernel_size=1)
    # run average pooling
    output = slim.avg_pool2d(inputs=output, kernel_size=2, stride=2)
    return output


def densenet_169(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 scope='DenseNet_169'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        weights_initializer=slim.variance_scaling_initializer()):
        with slim.arg_scope([slim.batch_norm],
                            decay=0.9997,
                            epsilon=0.001,
                            is_training=is_training,
                            updates_collections=tf.GraphKeys.UPDATE_OPS):
            with slim.arg_scope([slim.dropout], keep_prob=dropout_keep_prob):

                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

                    net = densenet_base(inputs, growth_rate_k=12, block_list=[6, 12, 32, 32], bc_mode=True)

                    net = slim.batch_norm(net)
                    net = tf.nn.relu(net)

                    net = slim.avg_pool2d(net, [7, 7], stride=1, scope='AvgPool_0a_7x7')
                    from tensorflow.contrib.slim import conv2d
                    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

                    # FC
                    features_total = int(net.get_shape()[-1])
                    with tf.variable_scope("logits"):
                        weights = tf.get_variable("weights", shape=[features_total, num_classes],
                                                  initializer=tf.contrib.layers.xavier_initializer())
                        bias = tf.get_variable("bias", initializer=tf.constant(0.0, shape=[num_classes]))
                        logits = tf.matmul(net, weights) + bias

                    return logits

densenet_169.default_image_size = 224
