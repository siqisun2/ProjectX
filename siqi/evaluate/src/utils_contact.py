import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
import string
import sys
from datetime import datetime, timedelta
import importlib
import time
import cPickle 
import cPickle as pickle

def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)

def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())
    
def hostname():
    return platform.node()
    
def generate_expid(arch_name):
    return "%s-%s-%s" % (arch_name, hostname(), timestamp())


def current_learning_rate(schedule, idx):
    s = schedule.keys()
    s.sort()
    current_lr = schedule[0]
    for i in s:
        if idx >= i:
            current_lr = schedule[i]

    return current_lr



## calculate the confusion matrix
def confusionMatrix(pre_out, y, mask_2d, response="12C"):
    # pre_out has the shape (N, L, L)
    # y has the shape (N, L, L)
    # mask_2d has the shape (N, L, L)

    ## both pred and truth have shape (batchSize, seqLen, seqLen), so is z
    def confusionMatrix12C(pre_out, y):
        ##convert pred, truth to 3C
        truth1 = T.gt(y, 3)
        truth2 = T.gt(y,10)
        truth_new = T.cast( truth1 + truth2, 'int32')

        pred1 = T.gt(pre_out, 3)
        pred2 = T.gt(pre_out, 10)
        pred_new = T.cast( pred1 + pred2, 'int32')

        return confusionMatrix3C(pred_new, truth_new)

    def confusionMatrix3C(pre_out, y):

        selMatrix = T.ones_like(y)
        if mask_2d is not None:
            mshape = mask_2d.shape
            selMatrix = T.set_subtensor(selMatrix[:, 0:mshape[1], 0:mshape[2]], mask_2d)
            #selMatrix = T.set_subtensor(selMatrix[:, :, :mshape[1]], mask_2d.dimshuffle(0,2,1) )

            dataLen = y.shape[1]
            triu_3d_matrices = []
        for sep in [24, 12, 6]:
            triu_3d_matrices.append( T.triu(T.ones((dataLen, dataLen), dtype=np.int32), sep).dimshuffle('x', 0, 1) )

        triu_LR = triu_3d_matrices[0]
        triu_MR = triu_3d_matrices[1] - triu_3d_matrices[0]
        triu_SR = triu_3d_matrices[2] - triu_3d_matrices[1]

        pred_truth = y * 3 + pre_out

        confMs = []
        for t in [ triu_LR, triu_MR, triu_SR ]:
            sel = T.mul( t, selMatrix)
            pred_truth_selected = pred_truth[ sel.nonzero() ]
            predcount = T.bincount( pred_truth_selected, minlength=9 ).reshape((3,3))
            confMs.append(predcount)

        return T.stacklists( confMs )

    if response == '12C':
        return confusionMatrix12C(pre_out, y)
    elif response == '3C':
        return confusionMatrix3C(pre_out, y)
    else:
        print 'unsupported response: ', response
        sys.exit(-1)

## this function returns the long-, medium- and short-range prediction accuracy when proteinLen*topRatio predicted contacts are evaluated
## note that the average accuracy is only for one batch. To calculate the avg accuracy of multiple batches, u may have to take into consideration
## the fact that two batches may have different numbers of proteins
## this function returns a 4*3 matrix where rows 0, 1, and 2, 4 correspond to long-, medium-, medium-long- and short-range accuracy.
## column 0 is the accuracy, column 1 is the average number of predicted contacts and column 2 is the number of truth contacts among all the top proteinLen*topRatio predicted contacts
## the average number of predicted contacts may not be equal to avg(proteinLen)* topRatio when topRatio is big and proteinLen is small
def TopAccuracyByRange(pre_out_prob, y, mask_2d, response="12C", topRatio=0.5):
    # pre_out has the shape (N, L, L, 12)
    # y has the shape (N, L, L)
    # mask_1d has the shape (N, L, L)
    # weight has the shape (N, L, L)

    ## in this function, we assume that both pred is tensor3 and truth is a matrix
    ## the pred tensor contains predicted probability for each residue pair
    ## pred has shape (dataLen, dataLen, 12)
    ## truth has shape (dataLen, dataLen)
    def TopAccuracy12C(pred=None, truth=None):
        ## convert pred and truth to 3C
        ## 0 for 0,1,2,3
        ## 1 for 4,5,...,10
        ## 2 for 11
        truth1 = T.gt(truth, 3)
        truth2 = T.gt(truth, 10)
        truth_new = T.cast( truth1 + truth2, 'int32')

        pred1 = T.sum(pred[:, :, 0:4], axis=2, keepdims=True)
        pred2 = T.sum(pred[:, :, 4:11], axis=2, keepdims=True)
        pred3 = pred[:, :, 11].dimshuffle(0, 1, 'x')
        pred_new = T.concatenate( (pred1, pred2, pred3), axis=2)

        return TopAccuracy3C(truth=truth_new, pred=pred_new)


    ## in this function, we assume that pred is tensor3 and truth is a matrix
    ## pred has shape (dataLen, dataLen, 3) and truth has shape (dataLen, dataLen)
    def TopAccuracy3C(pred=None, truth=None):

        M1s = T.ones_like(truth, dtype=np.int8)
        LRsel = T.triu(M1s, 24)
        MLRsel = T.triu(M1s, 12)
        SMLRsel = T.triu(M1s, 6)
        MRsel = MLRsel - LRsel
        SRsel = SMLRsel - MLRsel

        dataLen = truth.shape[0]

        avg_pred = (pred + pred.dimshuffle(1, 0, 2) )/2.0

        pred_truth = T.concatenate( (avg_pred, truth.dimshuffle(0, 1, 'x') ), axis=2)

        accuracyList = []
        for Rsel in [ LRsel, MRsel, MLRsel, SRsel]:
            selected_pred_truth = pred_truth[Rsel.nonzero()]

            ## sort by the predicted probability for label 0, which inidicates contact
            selected_pred_truth_sorted = selected_pred_truth[ (selected_pred_truth[:,0]).argsort()[::-1] ]

            #print 'topRatio =', topRatio
            numTops = T.minimum( T.iround(dataLen * topRatio), selected_pred_truth_sorted.shape[0])

            selected_sorted_truth = T.cast(selected_pred_truth_sorted[:, 3], 'int32')
            numTruths = T.bincount(selected_sorted_truth, minlength=3 )
            numCorrects = T.bincount( selected_sorted_truth[0:numTops], minlength=3 )
            #numTops = T.minimum(numTops, numTruths[0])
            accuracyList.append( T.stack( [numCorrects[0] *1./(numTops + 0.001), numTops, numTruths[0] ], axis=0)  )

        return T.stacklists( accuracyList )

    def EvaluateAccuracy(pred_prob, truth, pad_len):
        pred_in_correct_shape = T.cast( pred_prob[0:pad_len, 0:pad_len, :], dtype=theano.config.floatX)
        truth_in_correct_shape = T.cast( truth[0:pad_len, 0:pad_len], 'int32')
        if response == '12C':
            return TopAccuracy12C(pred_in_correct_shape, truth_in_correct_shape)
        elif response == '3C':
            return TopAccuracy3C(pred_in_correct_shape, truth_in_correct_shape)
        else:
            raise NotImplementedError

    mask_1d = mask_2d[:, 0, :]
    ## we need a scan operation here
    if mask_2d is not None:
        paddingLens = T.sum(mask_1d, axis=1)
    else:
        paddingLens = T.zeros_like(y[:,0,0], dtype=np.int32)

    #result, updates = theano.scan(fn=EvaluateAccuracy, outputs_info=None, sequences=[self.output_2d_prob, z, paddingLens] )
    result, updates = theano.scan(fn=EvaluateAccuracy, outputs_info=None, sequences=[pre_out_prob, y, paddingLens] )
    accuracy = T.mean(result, axis=0)

    return accuracy

def TopAccuracy(pred=None, truth=None, ratio=[2, 1, 0.5, 0.2, 0.1], responseStr='12C'):
    if pred is None:
        print 'please provide a predicted contact matrix'
        sys.exit(-1)

    if truth is None:
        print 'please provide a true contact matrix'
        sys.exit(-1)

    assert pred.shape[0] == pred.shape[1]
    assert pred.shape == truth.shape

    pred_truth = np.dstack( (pred, truth) )

    M1s = np.ones_like(truth, dtype = np.int8)
    mask_LR = np.triu(M1s, 24)
    mask_MLR = np.triu(M1s, 12)
    mask_SMLR = np.triu(M1s, 6)
    mask_MR = mask_MLR - mask_LR
    mask_SR = mask_SMLR - mask_MLR

    seqLen = pred.shape[0]

    accs = []
    for mask in [ mask_LR, mask_MR, mask_MLR]:

        res = pred_truth[mask.nonzero()]
        res_sorted = res [ (-res[:,0]).argsort() ]

        for r in ratio:
            numTops = int(seqLen * r)
            numTops = min(numTops, res_sorted.shape[0] )
            topLabels = res_sorted[:numTops, 1]
            if responseStr == '12C':
                numCorrects = ( (0<=topLabels) & (topLabels<4) ).sum()
            else:
                numCorrects = ( (0<=topLabels) & (topLabels<=0) ).sum()
            accuracy = numCorrects * 1./numTops
            accs.append(accuracy)

    return np.array(accs)