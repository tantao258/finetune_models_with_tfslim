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

from nets import dense_utils

slim = tf.contrib.slim


def densenet_base(inputs,
                  growth_rate_k=32,
                  block_list=[6, 12, 32, 32],
                  final_endpoint="",
                  scope='DenseNet',
                  bc_mode=False,
                  reduction=1.0,
                  ):
    """Defines the DenseNet base architecture."""

    end_points = {}
    with tf.variable_scope(scope, 'DenseNet', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):

            end_point = 'Conv2d_1a_7x7'
            net = slim.conv2d(inputs, growth_rate_k * 2, [7, 7], stride=2, scope=end_point)
            end_points[end_point] = net
            if final_endpoint == end_point:
                return net, end_points

            end_point = 'MaxPool_1b_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if final_endpoint == end_point:
                return net, end_points

            for i, layer_num in enumerate(block_list, start=1):

                with tf.variable_scope("Block_%d" % i):
                    end_point = 'Bolck_%d' % i

                    net = add_block(net, growth_rate_k, layer_num, bc_mode)

                    end_points[end_point] = net
                    if final_endpoint == end_point:
                        return net, end_points

                # last block exist without transition layer
                if i != len(block_list) - 1:
                    with tf.variable_scope("Transition_after_block_%d" % i):
                        end_point = "Transition_after_block_%d" % i

                        net = transition_layer(net, reduction)

                        end_points[end_point] = net
                        if final_endpoint == end_point:
                            return net, end_points

            net = slim.batch_norm(net)
            net = tf.nn.relu(net)

            return net, end_points


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
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='DenseNet_169',
                 global_pool=False):
    """Defines the densenet_169 architecture.

    This architecture is defined in:

      Densely Connected Convolutional Networks
      Gao Huang, Zhuang Liu, Laurens van der Maaten
      https://arxiv.org/pdf/1608.06993.pdf.

    The default image size used to train this network is 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before dropout)
        are returned instead.
      is_training: whether is training or not.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
          shape [B, 1, 1, C], where B is batch_size and C is number of classes.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
      global_pool: Optional boolean flag to control the avgpooling before the
        logits layer. If false or unset, pooling is done with a fixed window
        that reduces default-sized inputs to 1x1, while larger inputs lead to
        larger outputs. If true, any input size is pooled down to 1x1.

    Returns:
      net: a Tensor with the logits (pre-softmax activations) if num_classes
        is a non-zero integer, or the non-dropped-out input to the logits layer
        if num_classes is 0 or None.
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """

    # Final pooling and prediction
    with tf.variable_scope(scope, 'DenseNet_169', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            with slim.arg_scope([slim.dropout], keep_prob=dropout_keep_prob):
                net, end_points = densenet_base(inputs,
                                                growth_rate_k=32,
                                                block_list=[6, 12, 32, 32],
                                                scope=scope,
                                                bc_mode=True)

                with tf.variable_scope('Logits'):
                    if global_pool:
                        # Global average pooling.
                        net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
                        end_points['global_pool'] = net
                    else:
                        # Pooling with a fixed kernel size.
                        net = slim.avg_pool2d(net, [7, 7], stride=1, scope='AvgPool_0a_7x7')
                        end_points['AvgPool_0a_7x7'] = net

                    if not num_classes:
                        return net, end_points
                    logits = slim.conv2d(inputs=net, num_outputs=num_classes, kernel_size=1, scope='Conv2d_0c_1x1')

                    if spatial_squeeze:
                        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                    end_points['Logits'] = logits
                    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points

densenet_169.default_image_size = 224
densenet_169_arg_scope = dense_utils.densenet_arg_scope
