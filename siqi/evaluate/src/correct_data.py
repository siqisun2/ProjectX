import numpy as np
import cPickle
import theano.tensor as T
import theano
import os
import os.path
import sys
import time
import datetime
import random
import gzip

############## Define the modelSpecs #############
data_folder = '/home/siqi/Contact/data/raw/'

modelSpecs = {
 'L2reg': 9.9999997e-05,
 'LRbias': 'mid',
 'RefProbability': (np.array([[ 0.01159078,  0.09494101,  0.8934682 ],
         [ 0.03384808,  0.20579764,  0.76035428],
         [ 0.0480072 ,  0.40372956,  0.54826325],
         [ 0.53122324,  0.3524473 ,  0.11632945]], dtype=np.float32),
  np.array([[ 0.00162035,  0.00244012,  0.0032115 ,  0.00431882,  0.00614039,
           0.00925802,  0.01198427,  0.01356141,  0.01521944,  0.01802076,
           0.02075673,  0.8934682 ],
         [ 0.0048917 ,  0.00743198,  0.0093627 ,  0.01216171,  0.01534673,
           0.02188426,  0.02810171,  0.03067251,  0.03255296,  0.03659835,
           0.04064113,  0.76035428],
         [ 0.00709255,  0.01072007,  0.01286193,  0.01733266,  0.02317664,
           0.03995903,  0.08614379,  0.06634221,  0.06531696,  0.05648016,
           0.06631076,  0.54826325],
         [ 0.05674842,  0.23028296,  0.11707992,  0.12711194,  0.05771757,
           0.08798206,  0.07017874,  0.04719847,  0.03295555,  0.03140707,
           0.02500785,  0.11632945]], dtype=np.float32)),
 'SeparateTrainByRange': False,
 'UsePSICOV': False,
 'UsePriorDistancePotential': False,
 'UseSampleWeight': True,
 'activation': T.nnet.nnet.relu,
 'algorithm': 'Adam',
 'batchNorm': True,
 'conv1d_hiddens': [50, 50],
 'conv1d_repeats': [1, 0],
 'conv2d_hiddens': [50, 55, 60, 65, 70, 75],
 'conv2d_repeats': [4, 4, 4, 4, 4, 3],
 'halfWinSize_matrix': 2,
 'halfWinSize_seq': 7,
 'logreg_hiddens': [60],
 'maxbatchSize': 122500,
 'minibatchSize': 60000,
 'mode': 'SMLR',
 'n_in_matrix': 5,
 'n_in_seq': 26,
 'numUpdates': 27,
 'predFile_CASP': [data_folder + 'CASP.contactFeatures.pkl'],
 'predFile_CAMEO': [data_folder + 'CAMEO.contactFeatures.pkl'],
 'response': '12C',
 'seq2matrixMode': {'OuterCat': [70, 35], 'Seq+SS': [4, 6, 12]},
 'trainFile': [data_folder + 'pdb25-train-P0.contactFeatures.pkl',
  data_folder + 'pdb25-train-P1.contactFeatures.pkl',
  data_folder + 'pdb25-train-P2.contactFeatures.pkl',
  data_folder + 'pdb25-400to500.contactFeatures.pkl',
  data_folder + 'pdb25-test-P2.contactFeatures.pkl'],
 'type': 'ResNetv4',
 'validFile': [data_folder + 'pdb25-test-P0.contactFeatures.pkl',
  data_folder + 'pdb25-test-P1.contactFeatures.pkl'],
 'weight412C':
  np.array([[ 36.66054916,  24.34420395,  18.49689293,  13.75440311,
          11.92761421,   7.91100264,   6.11136436,   5.40063572,
           4.81228018,   4.06421328,   3.52850485,   1.        ],
        [  9.34131336,   6.14841747,   4.8805294 ,   3.75727773,
           3.62066507,   2.53905678,   1.9772948 ,   1.81156909,
           1.70692182,   1.51824772,   1.36721992,   1.        ],
        [  4.90729523,   3.24673605,   2.70606589,   2.00807166,
           1.74196792,   1.01035881,   0.46866935,   0.60855609,
           0.61810833,   0.71481657,   0.60884476,   1.        ]], dtype=np.float32),
 'weight43C': 
 np.array([[ 20.5       ,   5.4000001 ,   1.        ],
        [  5.4000001 ,   1.88999999,   1.        ],
        [  2.9000001 ,   0.69999999,   1.        ]], dtype=np.float32),
 'weight4range': np.array([ 3.,  3.,  1.], dtype=np.float32)}


def RowWiseOuterProduct(A, B):
    a = A[:, :, np.newaxis ]
    b = B[:, np.newaxis, : ]
    c = a * b
    return c.reshape( (A.shape[0], A.shape[1] * B.shape[1]) )

def PriorDistancePotential(sequence=None, paramfile=None):
    ##add pairwise distance potential here
    #fh = open('data4contact/pdb25-pair-dist.pkl','rb')
    if not os.path.isfile(paramfile):
        print 'cannot find the parameter file: ', paramfile
        sys.exit(-1)

    fh = open(paramfile,'rb')
    potential = cPickle.load(fh)[0].astype(np.float32)
    fh.close()
    #sequence = d['sequence'].upper()
    potentialFeature = np.zeros((len(sequence), len(sequence), 12), dtype=np.float32)

    for i, AAi in zip(xrange(len(sequence)), sequence):
        for j, AAj in zip(xrange(i+1, len(sequence)), sequence[i+1:]):
            if j-i<6:
                sepIndex = 0
            elif j-i < 12:
                sepIndex = 1
            elif j-i < 24:
                sepIndex = 2
            else:
                sepIndex = 3

            id0 = ord(AAi)-ord('A')
            id1 = ord(AAj)-ord('A')
            if id0 > id1:
                id1 = ord(AAi)-ord('A')
                id0 = ord(AAj)-ord('A')

            potentialFeature[i][j]=potential[sepIndex][id0][id1]
            potentialFeature[j][i]=potentialFeature[i][j]
    return potentialFeature

def DistMatrix2Contact(distm, response):
    if response == '6CSum':
        from Labels import CalcLabelMatrix_6CSum
        labelMatrix = CalcLabelMatrix_6CSum(distm)
        return labelMatrix

    contactMap = np.zeros( distm.shape, dtype=np.int8 )
    if response == '12C':
        ## discretize distance into 12 intervals at 11 points: 5, 6, 7, 8, 9 10, 11, 12, 13, 14, 15, >15 or <0
        np.putmask(contactMap, distm<0., 11)
        np.putmask(contactMap, distm>5, 1)
        np.putmask(contactMap, distm>6, 2)
        np.putmask(contactMap, distm>7, 3)
        np.putmask(contactMap, distm>8, 4)
        np.putmask(contactMap, distm>9, 5)
        np.putmask(contactMap, distm>10, 6)
        np.putmask(contactMap, distm>11, 7)
        np.putmask(contactMap, distm>12, 8)
        np.putmask(contactMap, distm>13, 9)
        np.putmask(contactMap, distm>14, 10)
        np.putmask(contactMap, distm>15, 11)
    else:
    ## discretize distance into 3 intervals: [0-8], [8-15], >15
        np.putmask(contactMap, distm>8, 1)
        np.putmask(contactMap, distm>15, 2)
        np.putmask(contactMap, distm<0., 2)
    return contactMap

def LocationFeature(d):
    ##add one specific location feature here, i.e., posFeature[i, j]=min(1, abs(i-j)/24.0 )
    posFeature = np.ones_like(d['ccmpredZ']).astype(np.float32)
    separation_cutoff = 30
    end = min(separation_cutoff - 1, posFeature.shape[0])
    for offset in xrange(0, end):
        i = np.arange(0, posFeature.shape[0]-offset)
        j = i + offset
        posFeature[i, j] = offset/(1. * separation_cutoff)
    for offset in xrange(1, end):
        i = np.arange(offset, posFeature.shape[0])
        j = i - offset
        posFeature[i, j] = offset/(1. * separation_cutoff)
    return posFeature



def LoadContactFeatures(files=None, modelSpecs=None):
    if files is None or len(files)==0:
        print 'the feature file is empty'
        sys.exit(-1)

    fhs = [ open(file, 'rb') for file in files ]
    data = sum([ cPickle.load(fh) for fh in fhs ], [])
    [ fh.close() for fh in fhs ]

    ## for each protein, we have one sequential feature, one matrix feature and one contact map as label

    proteinFeatures=[]
    counter = 0
    for d in data:

        oneprotein = dict()
        oneprotein['name'] = d['name']
        if d.has_key('otherSeqFeatures'):
            seqFeature = np.concatenate( (d['SS3'], d['ACC'], d['PSSM'], d['otherSeqFeatures']), axis=1).astype(np.float32)
        else:
            seqFeature = np.concatenate( (d['SS3'], d['ACC'], d['PSSM']), axis=1).astype(np.float32)

        if d.has_key('ccmpredZ') is not True:
            print 'Something must be wrong. The data does not have ccmpred feature!'
            sys.exit(-1)

        ##add one specific location feature here, i.e., posFeature[i, j]=min(1, abs(i-j)/24.0 )
        posFeature = LocationFeature(d)

        pairfeatures = [ posFeature, d['ccmpredZ'], d['OtherPairs'] ]

        ##add pairwise distance potential here
        if modelSpecs['UsePriorDistancePotential'] is True:
            INSTALLDIR = os.getenv('gcnnContactPredHome')
            if INSTALLDIR is None:
                print 'please set the environment variable gcnnContactPredHome as the installation directory of the contact prediction program'
                sys.exit(-1)
            if not INSTALLDIR.endswith('/'):
                INSTALLDIR += '/'

            potentialFeature = PriorDistancePotential(sequence=d['sequence'].upper(), paramfile=(INSTALLDIR + 'data4contact/pdb25-pair-dist.pkl') )
            pairfeatures.append(potentialFeature)

        if modelSpecs['UsePSICOV'] is True:
            pairfeatures.append(d['psicovZ'])

        matrixFeature = np.dstack( tuple(pairfeatures) )
        #print 'matrixFeature.shape: ', matrixFeature.shape

        oneprotein['sequence'] = d['sequence']
        oneprotein['seqLen'] = seqFeature.shape[0]
        oneprotein['seqFeatures'] = seqFeature
        oneprotein['matrixFeatures'] = matrixFeature

        if d.has_key('DistMatrix'):
            distm = d['DistMatrix']
            response = modelSpecs['response']
            contactMap = DistMatrix2Contact(distm, response)
            oneprotein['contactMap'] = contactMap
            oneprotein['distMatrix'] = d['DistMatrix']

        proteinFeatures.append(oneprotein)

        counter += 1
        if (counter %500 ==1):
            print 'assembled features for ', counter, ' proteins.'

    return proteinFeatures, data

def SelectionMatrix(ContactMatrix=None, dataLen=10, modelSpecs=None, DistMatrix=None):
    if ContactMatrix is not None:
        size = ContactMatrix.shape
    elif dataLen > 24:
        size = (dataLen, dataLen)
    else:
        print 'Do not know how to determine the size of the selection matrix or the protein is shorter than 25 AAs'
        sys.exit(-1)

    mode = modelSpecs['mode']
    if mode == 'LR':
        separation = 24
    elif mode == 'MLR':
        separation = 12
    elif mode == 'SMLR':
        separation = 6
    else:
        print 'unsupported mode for selection matrix generation: ', mode
        sys.exit(-1)

    M1s = np.ones(size, dtype=np.int16)

    ##currently we consider all residue pairs
    ##in the future we may do sampling
    selMatrix = np.triu(M1s, separation) + np.tril(M1s, -separation)

    ## for prediction, we do not have contact matrix. In this case, we simply return the selection matrix
    if ContactMatrix is None:
        return selMatrix.astype(theano.config.floatX)

    ## now we may add weight to each element in the selection matrix
    if modelSpecs['response'] == '3C':
        wMatrix = modelSpecs['weight43C']
    elif modelSpecs['response' ] == '12C':
        wMatrix = modelSpecs['weight412C']
    elif modelSpecs['response'] == '6CSum':
        wMatrix = modelSpecs['weight46CSum']
    else:
        raise NotImplementedError

    threeWeightMatrices=[]
    for i in xrange(3):
        threeWeightMatrices.append( np.choose(ContactMatrix, wMatrix[i] * modelSpecs['weight4range'][i]) )

    ##assemble the above thee matrices into a single one
    LRmask = np.triu( M1s, 24) + np.tril( M1s, -24)
    MLRmask = np.triu( M1s, 12) + np.tril( M1s, -12)
    SRmask = np.triu( M1s, 6) + np.tril(M1s, -6) - MLRmask
    MRmask = MLRmask - LRmask

    LRw, MRw, SRw = threeWeightMatrices

    selMatrix = selMatrix * (LRmask * LRw + MRmask* MRw + SRmask * SRw)
    
    #selMatrix[ DistMatrix == 999 ] = 0  #remove all disorder regions

    return selMatrix.astype(theano.config.floatX)

##split data into minibatch, each minibatch numDataPoints data points
def PrepareData(data=None, numDataPoints=1000000, modelSpecs=None, sort=True, weighted=True):

    if data is None:
        print 'Please provide a valid data for process!'
        sys.exit(-1)

    if numDataPoints < 10:
        print 'Please specify the number of data points in a minibatch'
        sys.exit(-1)


    ## sort from large to small
    if sort == True:
        data.sort(key=lambda x: x['seqLen'], reverse=True)
    seqDataset = []

    i = 0
    while i < len(data):
        currentSeqLen = data[i]['seqLen']
        numSeqs = min( len(data) - i, numDataPoints/(currentSeqLen * currentSeqLen) )
        numSeqs = max(1, numSeqs)
        #print 'This batch contains ', numSeqs, ' sequences'

        seqLens = [ d['seqLen'] for d in data[i: i + numSeqs ] ]
        maxSeqLen = max( seqLens )
        #minSeqLen = min( seqLens )
        #print 'maxSeqLen= ', maxSeqLen  #, 'minSeqLen= ', minSeqLen

        X1d = np.zeros(shape=(numSeqs, maxSeqLen, data[i]['seqFeatures'].shape[1] ), dtype = theano.config.floatX)
        X2d = np.zeros(shape=(numSeqs, maxSeqLen, maxSeqLen, data[i]['matrixFeatures'].shape[2] ), dtype = theano.config.floatX)
        #Sel = np.zeros(shape=(numSeqs, maxSeqLen, maxSeqLen), dtype = np.int16)
        Sel = np.zeros(shape=(numSeqs, maxSeqLen, maxSeqLen), dtype = theano.config.floatX)

        ## 1d features for embedding
        n_embed_in = None
        if modelSpecs['seq2matrixMode'].has_key('Seq+SS'):
            n_embed_in = 26 * 3
        elif modelSpecs['seq2matrixMode'].has_key('SeqOnly'):
            n_embed_in = 26

        X1dem = None
        if n_embed_in is not None:
            X1dem = np.zeros(shape=(numSeqs, maxSeqLen, n_embed_in ), dtype = theano.config.floatX)

        Y = None
        Y_dist = None
        if data[i].has_key('contactMap'):
            Y = np.zeros(shape=(numSeqs, maxSeqLen, maxSeqLen), dtype = np.int8 )
            Y_dist = np.zeros(shape=(numSeqs, maxSeqLen, maxSeqLen), dtype = theano.config.floatX )

        M1d = np.zeros(shape=(numSeqs, maxSeqLen), dtype=np.int8 )
        M2d = np.zeros(shape=(numSeqs, maxSeqLen, maxSeqLen), dtype=np.int8 )

        for j in xrange(numSeqs):
            dataLen = data[i+j]['seqLen']

            X1d[j, 0:dataLen, : ] = data[i+j]['seqFeatures']
            X2d[j, 0:dataLen, 0:dataLen, : ] = data[i+j]['matrixFeatures']

            if weighted and data[i+j].has_key('contactMap'):
                Sel[j, 0:dataLen, 0:dataLen] = SelectionMatrix(ContactMatrix=data[i+j]['contactMap'], modelSpecs=modelSpecs,\
                                                               DistMatrix   =data[i+j]['distMatrix'])
            else:
                Sel[j, 0:dataLen, 0:dataLen] = SelectionMatrix(dataLen=dataLen, modelSpecs=modelSpecs)

            M1d[j, 0:dataLen].fill(1)
            M2d[j, 0:dataLen, 0:dataLen].fill(1)

            if X1dem is not None:
                seq1d = (np.array(map(ord, data[i+j]['sequence'])) - ord('A') ).astype(np.int32)
                seqMatrix = np.zeros((dataLen, 26))
                seqMatrix[xrange(dataLen), seq1d ] = 1

                if modelSpecs['seq2matrixMode'].has_key('Seq+SS'):
                    ssMatrix = data[i+j]['seqFeatures'][:,0:3]
                    embedFeature = RowWiseOuterProduct(seqMatrix, ssMatrix)
                elif modelSpecs['seq2matrixMode'].has_key('SeqOnly'):
                    embedFeature = seqMatrix

                X1dem[j, 0:dataLen, : ] = embedFeature

            if Y is not None:
                Y[j, 0:dataLen, 0:dataLen] = data[i+j]['contactMap']
                Y_dist[j, 0:dataLen, 0:dataLen] = data[i+j]['distMatrix']

        onebatch = [X1d, X2d, M1d, M2d, Sel]
        if X1dem is not None:
            onebatch.append(X1dem)

        if Y is not None:
            onebatch.append(Y)
            onebatch.append(Y_dist)

        seqDataset.append(onebatch)

        i += numSeqs

    return seqDataset

def PrepareData4Model(filename, groupSize=modelSpecs['minibatchSize'], sort = True):
    #print 'preparing data for', filename
    data, _ = LoadContactFeatures(filename, modelSpecs=modelSpecs)
    SeqDataset = PrepareData(data=data, numDataPoints=groupSize, modelSpecs=modelSpecs, sort=sort)
    return SeqDataset


def PrepareTrainData(weighted=True):
    trainData, rawData = LoadContactFeatures(modelSpecs['trainFile'], modelSpecs=modelSpecs )
    validData, _ = LoadContactFeatures(modelSpecs['validFile'], modelSpecs=modelSpecs )
    
    print 'Preparing data for training...'
    groupSize = modelSpecs['minibatchSize']
    trainSeqDataset = PrepareData(data=trainData, numDataPoints=groupSize, modelSpecs=modelSpecs, weighted=weighted)
    validSeqDataset = PrepareData(data=validData, numDataPoints=100, modelSpecs=modelSpecs, weighted=weighted)
    return trainSeqDataset, validSeqDataset

def PrepareTestData(weighted=True):
    CASPpredData, _ = LoadContactFeatures(modelSpecs['predFile_CASP'], modelSpecs=modelSpecs )
    CAMEOpredData, _ = LoadContactFeatures(modelSpecs['predFile_CAMEO'], modelSpecs=modelSpecs )
    
    CASPpredSeqDataset = PrepareData(data=CASPpredData, numDataPoints=100, modelSpecs=modelSpecs, weighted=weighted)
    CAMEOpredSeqDataset = PrepareData(data=CAMEOpredData, numDataPoints=100, modelSpecs=modelSpecs, weighted=weighted)
    return CASPpredSeqDataset, CAMEOpredSeqDataset

    
def main():
    
    ####### load train, validation and test dataset ############################

    trainData, rawData = LoadContactFeatures(modelSpecs['trainFile'], modelSpecs=modelSpecs )
    validData, _ = LoadContactFeatures(modelSpecs['validFile'], modelSpecs=modelSpecs )
    CASPpredData, _ = LoadContactFeatures(modelSpecs['predFile_CASP'], modelSpecs=modelSpecs )
    CAMEOpredData, _ = LoadContactFeatures(modelSpecs['predFile_CAMEO'], modelSpecs=modelSpecs )

    print '#trainData: ', len(trainData), '#validData: ', len(validData), \
          '#CASPpredData: ', len(CASPpredData), '#CAMEOpredData: ', len(CAMEOpredData),

    ###### obtain data for training #####
    print 'Preparing data for training...'
    groupSize = modelSpecs['minibatchSize']
    trainSeqDataset = PrepareData(data=trainData, numDataPoints=groupSize, modelSpecs=modelSpecs)
    validSeqDataset = PrepareData(data=validData, numDataPoints=100, modelSpecs=modelSpecs)
    
    CASPpredSeqDataset = PrepareData(data=CASPpredData, numDataPoints=100, modelSpecs=modelSpecs)
    CAMEOpredSeqDataset = PrepareData(data=CAMEOpredData, numDataPoints=100, modelSpecs=modelSpecs)
    del trainData, validData, CASPpredData, CAMEOpredData

    print "#trainData minibatches:", len(trainSeqDataset), "#validData minibatches:", len(validSeqDataset), \
          "#CASPpredData minibatches:", len(CASPpredSeqDataset), "#CAMEOpredData minibatches:", len(CAMEOpredSeqDataset),
        
    return trainSeqDataset, validSeqDataset, CASPpredSeqDataset, CAMEOpredSeqDataset

if __name__ == '__main__':
    main()

