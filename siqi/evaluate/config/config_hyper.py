import lasagne
import numpy as np
import sys

import lasagne
import numpy as np
import sys
import theano.tensor as T
import theano

import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, ConcatLayer, DenseLayer,
                            DropoutLayer, Pool2DLayer, GlobalPoolLayer,
                            NonlinearityLayer)
from lasagne.nonlinearities import rectify, softmax
try:
    from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
except ImportError:
    from lasagne.layers import BatchNormLayer
    

class MetaContactEmbeddingLayer(lasagne.layers.Layer):
    """
    lasagne.layers.special.MetaContactEmbeddingLayer(incoming, **kwargs)

    A layer that just transfer the sequence feature N*L*78 to matrix feature N*L*L*m.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    """
    def __init__(self, incoming, num_units1, num_units2, num_units3, W1=lasagne.init.GlorotUniform(), 
                 W2=lasagne.init.GlorotUniform(), W3=lasagne.init.GlorotUniform(), **kwargs):
        super(MetaContactEmbeddingLayer, self).__init__(incoming, **kwargs)

        self.num_units1 = num_units1
        self.num_units2 = num_units2
        self.num_units3 = num_units3
        num_inputs = int(np.prod(self.input_shape[2:]))

        self.W1 = self.add_param(W1, (num_inputs, num_inputs, self.num_units1), name="W1")
        self.W2 = self.add_param(W2, (num_inputs, num_inputs, self.num_units2), name="W2")
        self.W3 = self.add_param(W3, (num_inputs, num_inputs, self.num_units3), name="W3")
        

        #num_units_max = max(num_units1, num_units2, num_units3)  

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1], \
                max(self.num_units1, self.num_units2, self.num_units3))

    def get_output_for(self, input, **kwargs):

        # LRembedLayer = (ContactEmbeddingLayer(input, self.num_units3)).get_output_for()
        # MRembedLayer = (ContactEmbeddingLayer(input, self.num_units2)).get_output_for()
        # SRembedLayer = (ContactEmbeddingLayer(input, self.num_units1)).get_output_for()
        input2 = input.dimshuffle(0,2,1)


        LRembedLayer1=  T.tensordot(input, self.W3, axes=1)
        LRembedLayer2 = LRembedLayer1.dimshuffle(0, 3, 1, 2)
        LRembedLayer3 = T.batched_tensordot(LRembedLayer2, input2, axes=1)
        LRembedLayer = LRembedLayer3.dimshuffle(0, 2, 3, 1)

        MRembedLayer1=  T.tensordot(input, self.W2, axes=1)
        MRembedLayer2 = MRembedLayer1.dimshuffle(0, 3, 1, 2)
        MRembedLayer3 = T.batched_tensordot(MRembedLayer2, input2, axes=1)
        MRembedLayer = MRembedLayer3.dimshuffle(0, 2, 3, 1)

        SRembedLayer1=  T.tensordot(input, self.W1, axes=1)
        SRembedLayer2 = SRembedLayer1.dimshuffle(0, 3, 1, 2)
        SRembedLayer3 = T.batched_tensordot(SRembedLayer2, input2, axes=1)
        SRembedLayer = SRembedLayer3.dimshuffle(0, 2, 3, 1)
        
        out1 = [LRembedLayer, MRembedLayer, SRembedLayer]

        M1s = T.ones( (input.shape[1], input.shape[1]) )
        Sep24Mat = T.triu(M1s, 24) + T.tril(M1s, -24)
        Sep12Mat = T.triu(M1s, 12) + T.tril(M1s, -12)
        Sep6Mat = T.triu(M1s, 6) + T.tril(M1s, -6)
        LRsel = Sep24Mat.dimshuffle('x', 0, 1, 'x')
        MRsel = (Sep12Mat - Sep24Mat).dimshuffle('x', 0, 1, 'x')
        SRsel = (Sep6Mat - Sep12Mat).dimshuffle('x', 0, 1, 'x')

        selections = [LRsel, MRsel, SRsel]

        out2 = T.zeros((input.shape[0], input.shape[1], input.shape[1], \
                max(self.num_units1, self.num_units2, self.num_units3)), dtype=theano.config.floatX)

        for emLayer, sel in zip(out1, selections):
            l_n_out = emLayer.shape[3]
            out2 = T.inc_subtensor(out2[:, :, :, : l_n_out], T.mul(emLayer, sel) )
        return out2
    
class SeqtoMatrixLayer(lasagne.layers.Layer):
    """
    lasagne.layers.special.SeqtoMatrixFeature(incoming, **kwargs)

    A layer that just transfer the sequence feature Batch_size*L*m to matrix feature Batch_size*L*L*3m.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    """
    def __init__(self, incoming, **kwargs):
        super(SeqtoMatrixLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        seqLen = input.shape[1]
        x = T.mgrid[0:seqLen, 0:seqLen]
        y1 = x[0]
        y2 = (x[0] + x[1])/2
        y3 = x[1]

        input2 = input.dimshuffle(1, 0, 2)

        out1 = input2[y1]
        out2 = input2[y2]
        out3 = input2[y3]

        out = T.concatenate([out1, out2, out3], axis=3)
        final_out = out.dimshuffle(2, 0, 1, 3)

        return final_out

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0], input_shapes[1], input_shapes[1], 3*input_shapes[2])
    
    
def OneDResNet():
    l_in_1 = lasagne.layers.InputLayer(shape=(None, None, 26))
    l_in_2 = lasagne.layers.InputLayer(shape=(None, None, None, 5))
    l_in_3 = lasagne.layers.InputLayer(shape=(None, None, 78))


    # 2. 1D CNN for sequence feature, N*L*26 -> N*1*L*26
    l_in_1_reshape = lasagne.layers.ReshapeLayer(l_in_1, (l_in_1.input_var.shape[0], 1, l_in_1.input_var.shape[1], 26))
    # first 1D CNN, N*1*L*26 -> N*50*L*1 (2DCNN output, BN should before nonlinear)-> N*1*L*50 (Dimshuffle)
    l_conv_a_1D1 = lasagne.layers.dnn.Conv2DDNNLayer(incoming=l_in_1_reshape, num_filters=50, pad= (15//2, 0),
                                          filter_size=(15, 26), stride=1, nonlinearity=rectify)
    l_conv_a_1D1_bn = BatchNormLayer(l_conv_a_1D1)
    l_conv_a_1D1_bn_dimshuffle = lasagne.layers.DimshuffleLayer(l_conv_a_1D1_bn, (0, 3, 2, 1))
    # second 1D CNN, N*1*L*50 -> N*50*L*1 -> N*1*L*50
    l_conv_a_1D2 = lasagne.layers.dnn.Conv2DDNNLayer(incoming=l_conv_a_1D1_bn_dimshuffle, num_filters=50, pad= (15//2, 0),
                                          filter_size=(15, 50), stride=1, nonlinearity=rectify)
    l_conv_a_1D2_bn = BatchNormLayer(l_conv_a_1D2)       
    l_conv_a_1D2_bn_dimshuffle = lasagne.layers.DimshuffleLayer(l_conv_a_1D2_bn, (0,3,2,1))
    # third 1D CNN, N*1*L*50 -> N*50*L*1 -> N*L*50*1 -> N*L*50 
    l_conv_a_1D3 = lasagne.layers.dnn.Conv2DDNNLayer(incoming=l_conv_a_1D2_bn_dimshuffle, num_filters=50, pad= (15//2, 0),
                                          filter_size=(15, 50), stride=1, nonlinearity=rectify)
    l_conv_a_1D3_bn = BatchNormLayer(l_conv_a_1D3)       
    l_conv_a_1D3_dimshuffle = lasagne.layers.DimshuffleLayer(l_conv_a_1D3_bn, (0,2,1,3))
    # due to modification, here l_conv_a_1D2_reshape should be revised to l_conv_a_1D3_reshape
    l_conv_a_1D3_reshape = lasagne.layers.ReshapeLayer(l_conv_a_1D3_dimshuffle, (l_in_1.input_var.shape[0], l_in_1.input_var.shape[1], 50))

    # Sequence feature to matrix feature
    # N*L*50 -> N*L*L*150 -> N*150*L*L -> N*70*L*L -> N*35*L*L -> N*L*L*35
    l_matrix1 = SeqtoMatrixLayer(l_conv_a_1D3_reshape)
    l_matrix1_dimshuffle = lasagne.layers.DimshuffleLayer(l_matrix1, (0, 3, 1, 2))
    l_matrix1_compress1 = lasagne.layers.dnn.Conv2DDNNLayer(incoming=l_matrix1_dimshuffle, num_filters=70, pad= 'same',
                                       filter_size=(1, 1), stride=(1, 1), nonlinearity=rectify)
    l_matrix1_compress1_bn = BatchNormLayer(l_matrix1_compress1)
    l_matrix1_compress2 = lasagne.layers.dnn.Conv2DDNNLayer(incoming=l_matrix1_compress1_bn, num_filters=35, pad= 'same',
                                   filter_size=(1, 1), stride=(1, 1), nonlinearity=rectify)
    l_matrix1_compress2_bn = BatchNormLayer(l_matrix1_compress2)
    l_matrix1_compress2_dimshuffle = lasagne.layers.DimshuffleLayer(l_matrix1_compress2_bn, (0, 2, 3, 1))

    # 3. MetaContactEmbedding for X_in_3 to obtain N*L*L*12 --------------
    # This embedding is for pairwise feature generation
    # N*L*78 -> N*L*L*12
    l_matrix2 = MetaContactEmbeddingLayer(l_in_3, 12, 6, 4)

    # N*L*L*(12+35+5) -> N*L*L*52 -> N*52*L*L
    l_c_b = lasagne.layers.ConcatLayer([l_in_2, l_matrix1_compress2_dimshuffle, l_matrix2], axis=3)
    l_c_b_dimshuffle = lasagne.layers.DimshuffleLayer(l_c_b, (0,3,1,2))
    return l_in_1, l_in_2, l_in_3, l_c_b_dimshuffle