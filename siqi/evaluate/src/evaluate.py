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
import src.utils_contact as utils
#from utils_contact import confusionMatrix, TopAccuracyByRange
import matplotlib.pyplot as plt
import random
import os
import glob
import numpy as np
import src.correct_data as read_data
from src.utils_contact import TopAccuracy



def ReadData(task, responseStr):
    jinbodata_folder = '/mnt/home/siqi/NewContact/gcnn4contactprediction/data4contact'
    db = dict()
    ###########################
    #datasets
    ###########################
    db['casp_files'] = ['CASP11AC.2015.contactFeatures.pkl', 'CASP11AC.2015-E1.contactFeatures.pkl',
                  'CASP11AC.2016.contactFeatures.pkl', 'CASP11AC.2016-E1.contactFeatures.pkl']

    db['cameo76_files'] = ['76CAMAC.2015.contactFeatures.pkl', '76CAMAC.2015-E1.contactFeatures.pkl',
                     '76CAMAC.2016.contactFeatures.pkl', '76CAMAC.2016-E1.contactFeatures.pkl']

    db['mems400_files'] = ['Mems400.2015.contactFeatures.pkl', 'Mems400.2015-E1.contactFeatures.pkl',
                    'Mems400.2016.contactFeatures.pkl', 'Mems400.2016-E1.contactFeatures.pkl']

    ###########################
    #datasets truth dir
    ###########################
    db['casp_truthdir'] = '/mnt/data/RaptorXCommon/TrainTestData/Distance_Contact_TrainData/449_CASP_CAMEO_Testing_Data/449_distcb'
    db['casp_list'] = np.loadtxt( os.path.join(jinbodata_folder, 'CASP11.list'), dtype='S13' )

    db['cameo76_truthdir'] = '/mnt/data/RaptorXCommon/TrainTestData/Distance_Contact_TrainData/77_CAMEO_Testing_Data/77_distcb/'
    db['cameo76_list'] = np.loadtxt( os.path.join(jinbodata_folder, '76CAMEO.list'), dtype='S13')

    db['mems400_truthdir'] = '/mnt/data/RaptorXCommon/TrainTestData/Membrane_Protein/400_memprot_dataset/400_testdata_distcb'
    db['mems400_list'] = np.loadtxt( os.path.join(jinbodata_folder,'400Mems.list'), dtype='S13')

    #############################
    #read dataset
    #############################
    X = []
    for c in db[ task + '_files']:
        X.append( read_data.PrepareData4Model([os.path.join(jinbodata_folder, c)], groupSize=10, sort=False) )

    truth_cm = []
    for c in db[task + '_list']:
        mat = np.loadtxt( os.path.join( db[task + '_truthdir'], c+'.distcb'))
        contactMap = read_data.DistMatrix2Contact(mat, responseStr)
        truth_cm.append( contactMap.reshape(1, contactMap.shape[0], contactMap.shape[1]) )
        
    return X, truth_cm, db
        
    
def LoadConfig(config_name, deterministic=True):   
    config = importlib.import_module(config_name)
    #optimizer = config.optimizer
    #assert num_classes == config.num_classes

    #sys.path.append( os.path.dirname(config_name))
    #config = importlib.import_module( os.path.basename(config_name) )
    #optimizer = config.optimizer

    ###################################
    #re-build model based on config
    ###################################
    sym_y = T.itensor3('target_contact_map')
    sym_mask = T.itensor3('mask')
    sym_weight = T.tensor3('contact_weight')

    l_in_1, l_in_2, l_in_3, l_out = config.build_model()
    print '# of parameters is ', nn.layers.count_params(l_out)
    out_eval = nn.layers.get_output(l_out, deterministic=deterministic)
    #out_eval = nn.layers.get_output(l_out)

    eval = theano.function([l_in_1.input_var, l_in_2.input_var, l_in_3.input_var, sym_y, sym_mask,\
                        sym_weight], out_eval, on_unused_input='ignore', \
                           allow_input_downcast=True)
    return l_out, eval

def FinalPredict(eval, X,  which_protein, truth, responseStr, db, task, sym=False, num_classes=12):
    L = X[0][which_protein][0].shape[1]
    final_res = np.zeros( (1, L, L, num_classes), dtype = np.float32)
    
    for j,c in zip( xrange( len(X) ), db[ task + '_files']):
        x = X[j][which_protein]
        batch_size_test_casp = (x[0]).shape[0]
        test_casp_out = eval(x[0], x[1], x[5], truth, x[3], x[4])
        #print test_casp_out.shape
        #break
        if responseStr == '12C':
            test_casp_out_contact = np.sum(test_casp_out[:, :, :, 0:4], axis=3, keepdims=True)
        else:
            test_casp_out_contact = np.sum(test_casp_out[:, :, :, 0:1], axis=3, keepdims=True)

        if sym:
            test_casp_out_contact[0,:,:,0] = test_casp_out_contact[0,:,:,0] + \
            test_casp_out_contact[0,:,:,0].transpose()
            
        final_res += test_casp_out
    if responseStr == '12C':
        final_res_contact = np.sum(final_res[:, :, :, 0:4], axis=3, keepdims=True)[0,:,:,0]

    else:
        final_res_contact = np.sum(final_res[:, :, :, 0:1], axis=3, keepdims=True)[0,:,:,0]

    if sym:
        final_res_contact = final_res_contact + final_res_contact.transpose()
    return final_res_contact

def AverageModels(l_outs, X, truth_cm, evals, output_file, responseStr, db, task, num_classes=12, sym=False, progress=True):
    N_set = len(X)
    N_protein = len(X[0])
    N_digits = 15
    N_models = 1
    
    #assert len(l_outs) == len(model_files) and len(model_files) == len(evals)
    
    log_file = open(output_file, 'w', 0)
    acc_ave = np.zeros( (N_protein, N_digits), dtype=np.float32)
    acc_model = np.zeros((N_models, N_protein, N_digits), dtype=np.float32)

    all_res = []
    
    for i in xrange( N_protein):
        res_tmp = []
        
        if progress:
            sys.stdout.write("\r%d / %d" % (i+1, N_protein ) )
            sys.stdout.flush()
        L = X[0][i][0].shape[1]
        ave_result = np.zeros( (L, L) )
        for idx_e, e in enumerate(evals):
            model_res_contact = FinalPredict(e, X, i, truth_cm[i], responseStr, db, task, sym)
            acc_model[idx_e, i, :] = TopAccuracy( model_res_contact,
                                         truth_cm[i][0], responseStr=responseStr)
            ave_result += model_res_contact
            res_tmp.append(model_res_contact)
            
        all_res.append(res_tmp)
        acc_ave[i] = TopAccuracy( ave_result, truth_cm[i][0], responseStr=responseStr)
    
    #return acc_model, acc_ave
    for k in xrange(N_models):
        print >> log_file, ' '.join( [str(np.round(x,4)) for x in np.mean(acc_model, 1)[k]] )
    print >> log_file, ' '.join( [str(np.round(x,4)) for x in np.mean(acc_ave, 0)] )
    print >> log_file, '\n'
    return all_res


def EvalResult(l_out, model_files, eval, output_file, sym=False):
    #################################
    # average result
    #################################
    N_set = len(X)
    N_protein = len(X[0])
    N_digits = 15

    log_file = open(output_file, 'w', 0)

    for m in model_files:
        nn.layers.set_all_param_values(l_out, np.load(m)['param_values'])

        acc_ave = np.zeros( (N_protein, N_digits), dtype=np.float32)
        acc_all = np.zeros((N_set, N_protein, N_digits), dtype=np.float32)

        for i in xrange( N_protein):
            #sys.stdout.write("\r%d / %d" % (i+1, N_protein ) )
            #sys.stdout.flush()

            L = X[0][i][0].shape[1]
            final_res = np.zeros( (1, L, L, num_classes), dtype = np.float32)
            for j,c in zip( xrange( len(X) ), db[ task + '_files']):
                x = X[j][i]
                batch_size_test_casp = (x[0]).shape[0]
                test_casp_loss, test_casp_out, = eval(x[0], x[1], x[5], truth_cm[i], x[3], x[4])
                if responseStr == '12C':
                    test_casp_out_contact = np.sum(test_casp_out[:, :, :, 0:4], axis=3, keepdims=True)
                else:
                    test_casp_out_contact = np.sum(test_casp_out[:, :, :, 0:1], axis=3, keepdims=True)
                
                if sym:
                    test_casp_out_contact[0,:,:,0] = test_casp_out_contact[0,:,:,0] + \
                    test_casp_out_contact[0,:,:,0].transpose()
                acc_all[j, i, :] = TopAccuracy(test_casp_out_contact[0,:,:,0], truth_cm[i][0], \
                                               responseStr=responseStr)
                final_res += test_casp_out
            if responseStr == '12C':
                final_res_contact = np.sum(final_res[:, :, :, 0:4], axis=3, keepdims=True)[0,:,:,0]
               
            else:
                final_res_contact = np.sum(final_res[:, :, :, 0:1], axis=3, keepdims=True)[0,:,:,0]
            
            if sym:
                final_res_contact = final_res_contact + final_res_contact.transpose()
            acc_ave[i] = TopAccuracy( final_res_contact,
                                         truth_cm[i][0], responseStr=responseStr)
        print >> log_file, m
        for k in xrange(N_set):
            print >> log_file, ' '.join( [str(np.round(x,4)) for x in np.mean(acc_all, 1)[k]] )
        print >> log_file, ' '.join( [str(np.round(x,4)) for x in np.mean(acc_ave, 0)] )
        print >> log_file, '\n'