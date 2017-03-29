import lasagne
import numpy as np
import sys
import string
import matplotlib.pyplot as plt


def RunFuncs(x, y, w, func, response=1):
    batch_size_train = (x[0]).shape[0]
    loss_t, loss_t_cc, loss_t_w_cc, loss_reg, out_train = func(x[0], x[1], x[-1],y[response], x[3], w[response])
    contact_out_train = np.sum(out_train[:, :, :, 0:4], axis=3, keepdims=True)[:, :, :, 0]
    
    acc_all = []
    for b in xrange(batch_size_train):
        acc_all.append( TopAccuracy(contact_out_train[b, :, :], y[response][b]) )

    return np.array([loss_t, loss_t_cc.mean(), loss_t_w_cc, loss_reg] ),  \
           np.sum(np.multiply(x[3], w[response])), np.array(acc_all)
    
def VisualizeContact(real, pred, contact_only=True):
    fig = plt.figure(figsize=(12, 4))
    fig.add_subplot(121)
    plt.pcolor(pred/2+pred.transpose()/2)
    plt.colorbar()

    fig.add_subplot(122)
    if contact_only:
        plt.pcolor(real<=3)
    else:
        plt.pcolor(real)

    plt.colorbar()
    
def SummaryNet(l_out, num_para=True, print_shape=False):
    if num_para:
        print '# of parameters is', lasagne.layers.count_params(l_out)
    if print_shape:
        all_layers = lasagne.layers.get_all_layers(l_out)
        #print shape information for each layer
        for l in all_layers:
            name = string.ljust(l.__class__.__name__, 32)
            print "%s %s" %(name, lasagne.layers.get_output_shape(l))


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

def TruncateTrainData(X, Y, W, MAX_LEN=400):
    for i in range(len(X)):
        if X[i][0].shape[1] > MAX_LEN:
            X[i][0] = (X[i][0])[:, 0:MAX_LEN, :]
            X[i][1] = (X[i][1])[:, 0:MAX_LEN, 0:MAX_LEN, :]
            X[i][3] = (X[i][3])[:, 0:MAX_LEN, 0:MAX_LEN]
            X[i][4] = (X[i][4])[:, 0:MAX_LEN, :]   
            for j in xrange( len(Y[i])):
                Y[i][j] = (Y[i][j])[:, 0:MAX_LEN, 0:MAX_LEN]        
            for j in xrange( len(W[i])):
                W[i][j] = (W[i][j])[:, 0:MAX_LEN, 0:MAX_LEN]