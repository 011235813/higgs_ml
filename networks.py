import tensorflow as tf
from layers import *

def hidden3_bn(vec_input, n_hidden1, n_hidden2, n_hidden3, n_outputs, 
               nonlinearity3, is_train):

    h1 = batch_normalized_linear_layer(vec_input=vec_input,
                                       num_nodes=n_hidden1,
                                       nonlinearity=tf.nn.relu,
                                       is_train=is_train, scope='fc1')

    h2 = batch_normalized_linear_layer(vec_input=h1, num_nodes=n_hidden2,
                                       nonlinearity=tf.nn.relu,
                                       is_train=is_train, scope='fc2')

    h3 = batch_normalized_linear_layer(vec_input=h2, num_nodes=n_hidden3,
                                       nonlinearity=nonlinearity3,
                                       is_train=is_train, scope='fc3')    

    out = linear_layer(vec_input=h3, num_nodes=n_outputs,
                       nonlinearity=None, scope='out')

    return out
