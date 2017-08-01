import tensorflow as tf
import numpy as np

def batch_normalized_linear_layer(vec_input, num_nodes, nonlinearity,
                                  is_train, scope):
    if nonlinearity == None:
        nonlinearity = tf.identity

    with tf.variable_scope(scope):
        x = tf.contrib.layers.fully_connected(inputs=vec_input,
                                              num_outputs=num_nodes,
                                              activation_fn=None,
                                              scope='dense')
        y = tf.contrib.layers.batch_norm(inputs=x, center=True,
                                         scale=True,
                                         is_training=is_train, scope='bn')

    return nonlinearity(y)    


def linear_layer(vec_input, num_nodes, nonlinearity, scope):

    if nonlinearity == None:
        nonlinearity = tf.identity

    with tf.variable_scope(scope):
        h = tf.contrib.layers.fully_connected(inputs=vec_input,
                                              num_outputs=num_nodes,
                                              activation_fn=nonlinearity)

    return h
