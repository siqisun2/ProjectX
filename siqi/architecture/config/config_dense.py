import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, ConcatLayer, DenseLayer,
                            DropoutLayer, Pool2DLayer, GlobalPoolLayer,
                            NonlinearityLayer, DimshuffleLayer, ReshapeLayer)
from lasagne.nonlinearities import rectify, softmax
try:
    from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
except ImportError:
    from lasagne.layers import BatchNormLayer

    

import numpy as np
import sys
import theano.tensor as T
import theano
from config_hyper import OneDResNet

# helper function for projection_b
def ceildiv(a, b):
    return -(-a // b)

def DenseNet(incoming, l_in_1, classes=12, depth=40, first_output=16, growth_rate=12, num_blocks=3, dropout=0.0, filter_size = 3,\
             n_hidden1 = 40, n_hidden2 = 40):
    if (depth - 1) % num_blocks != 0:
        raise ValueError("depth must be num_blocks * n + 1 for some n")
    
    # input and initial convolution
    network = Conv2DLayer(incoming, first_output, filter_size, pad='same',  W=lasagne.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None, name='pre_conv')
    #print lasagne.layers.get_output_shape(network)

    n = (depth - 1) // num_blocks
    for b in range(num_blocks):
        network = dense_block(network, n - 1, growth_rate, dropout, 'block%d' % (b + 1), filter_size)
        if b < num_blocks - 1:
            network = transition(network, dropout, name_prefix='block%d_trs' % (b + 1))
        #print lasagne.layers.get_output_shape(network)
        
    fc1 = Conv2DLayer(network, num_filters=n_hidden1, pad= 'same', filter_size=(1, 1), stride=(1, 1), nonlinearity=rectify)
    fc1_bn = BatchNormLayer(fc1)
    fc2 = Conv2DLayer(fc1_bn, num_filters=n_hidden2, pad= 'same', filter_size=(1, 1), stride=(1,1), nonlinearity=rectify)
    fc2_bn = BatchNormLayer(fc2)

    fc2_shuffle = DimshuffleLayer(fc2_bn, (0, 2, 3, 1))
    out1 = ReshapeLayer(fc2_shuffle, (l_in_1.input_var.shape[0]*l_in_1.input_var.shape[1]\
                                     *l_in_1.input_var.shape[1], n_hidden2))

    out2 = DenseLayer(out1, num_units=classes, nonlinearity=softmax)
    #out3 = BatchNormLayer(out2)
    out = ReshapeLayer(out2, (l_in_1.input_var.shape[0], l_in_1.input_var.shape[1], l_in_1.input_var.shape[1], classes))

    #print lasagne.layers.get_output_shape(out)
    return out

        
def bn_relu_conv(network, channels, filter_size, dropout, name_prefix):
    #BN -> RELU -> CONV -> DROPOUT (OPTIONAL)
    network = BatchNormLayer(network, name=name_prefix + '_bn')
    network = NonlinearityLayer(network, nonlinearity=rectify, name=name_prefix + '_relu')
    network = Conv2DLayer(network, channels, filter_size, pad='same',
                          W=lasagne.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None,
                          name=name_prefix + '_conv')
    if dropout:
        network = DropoutLayer(network, dropout)
    return network

def dense_block(network, num_layers, growth_rate, dropout, name_prefix, filter_size):
    # concatenated 3x3 convolutions
    for n in range(num_layers):
        conv = bn_relu_conv(network, channels=growth_rate,
                            filter_size=filter_size, dropout=dropout, name_prefix=name_prefix + '_l%02d' % (n + 1))
        network = ConcatLayer([network, conv], axis=1, name=name_prefix + '_l%02d_join' % (n + 1))
    return network

def transition(network, dropout, name_prefix, pooling=False):
    # a transition 1x1 convolution followed by avg-pooling
    network = bn_relu_conv(network, channels=network.output_shape[1],
                           filter_size=1, dropout=dropout,
                           name_prefix=name_prefix)
    if pooling:
        network = Pool2DLayer(network, 2, mode='average_inc_pad', name=name_prefix + '_pool')
    return network

def build_model():
    l_in_1, l_in_2, l_in_3, l_1dout = OneDResNet()
    l_out = DenseNet(l_1dout, l_in_1, classes=12, depth=21, first_output=5, growth_rate=12, num_blocks=5, dropout=0.0,\
                    filter_size = 3, n_hidden1 = 1000, n_hidden2 = 1000)
    return l_in_1, l_in_2, l_in_3, l_out
