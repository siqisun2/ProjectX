import numpy as np
import theano
import theano.tensor as T
import string
import sys
from datetime import datetime, timedelta
import importlib
import time
import cPickle 
import cPickle as pickle
import matplotlib.pyplot as plt
import random
import os
import lasagne
import collections
import getopt



import config.config_dense as CD
import config.config_hyper as CH
import config.contact_util as CU

##### Default Hyper Parameters ############
optimizer = 'sgd'
init_lr = 0.001
reduce_lr = True 
dropout = 0.0

try:
    opts, args = getopt.getopt(sys.argv[1:], 'l:r:o:d:', ['lr=', 'reduce_lr=', 'optimizer=', 'dropout='])
except getopt.GetoptError:
    print 'option error'
    #Usage()
    sys.exit(-1)

if len(opts) < 2:
    #Usage()
    print 'option error'
    sys.exit(-1)
    
for opt, arg in opts:
    if opt in ('-d', '--dropout'):
        dropout = np.float(arg)
    if opt in ('-l', '--lr'):
        init_lr = np.float32( arg )
    if opt in ('-r', '--reduce_lr'):
        reduce_lr = np.bool( np.int(arg) )
    if opt in ('-o', '--optimizer'):
        optimizer = arg


TOL = 1e-5
start_epoch = 0
num_epochs = 36
MAX_LEN = 300
classes = 12
lambda_reg = 0.0001
cut_norm = 10
output_dir = '/mnt/home/siqi/x_models/dense_largest'
config_name = 'config_dense'
shuffle = True

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

experiment_id = "%s-%s-%s-%s-%s" % (config_name, optimizer, str(init_lr), str(reduce_lr), timestamp)
if dropout > 0.0:
    experiment_id = experiment_id + '-' + str(dropout)
metadata_path = os.path.join( output_dir, experiment_id)

response = 1 if classes == 12 else 0
print 'output path is', metadata_path, ', response is', response

##### Build Models ############
l_in_1, l_in_2, l_in_3, l_1dout = CH.OneDResNet()



##### Smaller Version
#l_out = CD.DenseNet(l_1dout, l_in_1, classes=classes, depth=19, first_output=4, growth_rate=12, num_blocks=2, dropout=0.0,\
#                    filter_size = 3, n_hidden1 = 40, n_hidden2 = 40)

##### Bigger Version
#l_out = CD.DenseNet(l_1dout, l_in_1, classes=classes, depth=36, first_output=5, growth_rate=12, num_blocks=5, dropout=dropout,\
#                    filter_size = 3, n_hidden1 = 200, n_hidden2 = 200)

l_out = CD.DenseNet(l_1dout, l_in_1, classes=classes, depth=37, first_output=15, growth_rate=12, num_blocks=4, dropout=dropout,\
                    filter_size = 3, n_hidden1 = 200, n_hidden2 = 200)


CU.SummaryNet(l_out, num_para=True, print_shape=False)

sym_y = T.itensor3('target_contact_map')
sym_mask = T.itensor3('mask')
sym_weight = T.tensor3('contact_weight')

grad_start_time = time.time()
out_train = lasagne.layers.get_output(l_out, deterministic=False)
out_eval = lasagne.layers.get_output(l_out, deterministic=True, batch_norm_use_averages=False)
pred_train = out_train[:, :, :, :].reshape((-1, classes))
pred_valid = out_eval [:, :, :, :].reshape((-1, classes))

params = lasagne.layers.get_all_params(l_out, regularizable=True)
reg_term = sum(T.sum(p**2) for p in params)

######### calculate cost function for train #######
cost_train_cc = lasagne.objectives.categorical_crossentropy(T.clip(pred_train, TOL, 1-TOL), sym_y.flatten())
cost_train_weight_cc = T.sum( cost_train_cc * sym_mask.flatten() * sym_weight.flatten() ) / \
                       T.sum( sym_weight.flatten() * sym_mask.flatten() )
cost_train = cost_train_weight_cc + lambda_reg * reg_term

######### calculate cost function for inference #######
cost_valid_cc = lasagne.objectives.categorical_crossentropy(T.clip(pred_valid, TOL, 1-TOL), sym_y.flatten())
cost_valid_weight_cc = T.sum(cost_valid_cc*sym_mask.flatten()*sym_weight.flatten()) / \
                       T.sum(sym_weight.flatten()*sym_mask.flatten())   # masked loss
cost_inference = cost_valid_weight_cc




##### Compile Models ############
sh_lr = theano.shared(lasagne.utils.floatX(init_lr))
all_params = lasagne.layers.get_all_params(l_out, trainable=True)

all_grads = [T.grad(cost_train, p, consider_constant= [sym_weight]) for p in all_params] 
print "Creating cost function and computing grads..."
updates, norm_calc = lasagne.updates.total_norm_constraint(all_grads, max_norm=cut_norm, return_norm=True)

if optimizer == 'adam':
    print 'Using Adam Optimizer, with init step size', sh_lr.get_value(),
    updates = lasagne.updates.adam(updates, all_params, learning_rate=sh_lr)
elif optimizer == 'sgd':
    print 'Using SGD Momentom Optimizer, with step size', sh_lr.get_value(),
    updates = lasagne.updates.nesterov_momentum(updates, all_params, sh_lr, 0.9)
print ', updates done, time cosuming', time.time() - grad_start_time, 's'

print 'compiling train and evals...'
t_compile = time.time()
train = theano.function([l_in_1.input_var, l_in_2.input_var, l_in_3.input_var, sym_y, sym_mask, sym_weight], \
                        [cost_train, cost_train_cc, cost_train_weight_cc, lambda_reg*reg_term, out_train], \
                        updates=updates, allow_input_downcast=True)

eval_valid  = theano.function([l_in_1.input_var, l_in_2.input_var, l_in_3.input_var, sym_y, sym_mask, sym_weight], \
                        [cost_inference, cost_valid_cc, cost_valid_weight_cc, lambda_reg*reg_term, out_eval], \
                        allow_input_downcast=True)

#eval_train  = theano.function([l_in_1.input_var, l_in_2.input_var, l_in_3.input_var, sym_y, sym_mask, sym_weight], \
#                        [cost_train, cost_train_cc, cost_train_weight_cc, lambda_reg*reg_term, out_train], \
#                        allow_input_downcast=True)
print "compile time %fs" %(time.time()-t_compile)




##### Load Data ############
print "loading data ...",
start_time = time.time()
feats = cPickle.load( open('/mnt/home/siqi/NewContact/TrainFeats/feats_train.pkl') )
weight = cPickle.load( open('/mnt/home/siqi/NewContact/TrainFeats/weights_train.pkl') )
contact = cPickle.load( open('/mnt/home/siqi/NewContact/TrainFeats/contact_train.pkl') )

x_train = feats['train_feat']; x_valid = feats['valid_feat']
w_train = weight['train_weight']; w_valid = weight['valid_weight']
y_train = contact['train_contact']; y_valid = contact['valid_contact']

feats = None; weight = None; contact = None
print "completed ..., it takes", time.time() - start_time, 's'

print 'Truncate data to MAX_LEN =', MAX_LEN
CU.TruncateTrainData(x_train, y_train, w_train, MAX_LEN = MAX_LEN)




##### Train ############
log_file = open( output_dir + '/log.' + experiment_id, 'w', 0)

loss_train_mean = []; acc_train_mean = []
loss_valid_mean = []; acc_valid_mean = []
loss_valid_mean2 = []; acc_valid_mean2 = []

for epoch in range(start_epoch, num_epochs):
    start_time = time.time()
    
    if shuffle:
        combined = list(zip(x_train, w_train, y_train))
        random.shuffle(combined)
        x_train[:], w_train[:],  y_train[:] = zip(*combined)
    ################   Train  ##########################
    loss_train_epoch = []; weight_train_epoch = [];  acc_train_epoch = []
    print 'epoch', epoch, 'with lr =', np.round( sh_lr.get_value(), 6)
    sys.stdout.flush
    
    for i in range(len(x_train)):
        sys.stdout.write('\r%d/%d for train'%(i+1, len(x_train) ))
        sys.stdout.flush()
        
        l, w, a = CU.RunFuncs(x_train[i], y_train[i], w_train[i], train)
        loss_train_epoch.append(l); weight_train_epoch.append(w); acc_train_epoch.append(a)       
    loss_train_mean.append( np.average(loss_train_epoch, axis=0, weights= weight_train_epoch) )
    acc_train_mean.append( np.mean( np.row_stack(acc_train_epoch), 0) )
    print '\t', ' '.join( [str(np.round(t,5)) for t in loss_train_mean[-1][[0,2,3]]] ), acc_train_mean[-1][4]
    
    ################   VALID  ##########################
    loss_valid_epoch = []; weight_valid_epoch = [];  acc_valid_epoch = []
    for i in range(len(x_valid)):
        sys.stdout.write('\r%d/%d for valid'%(i+1, len(x_valid) ))
        sys.stdout.flush()
        l, w, a = CU.RunFuncs(x_valid[i], y_valid[i], w_valid[i], eval_valid)
        loss_valid_epoch.append(l); weight_valid_epoch.append(w); acc_valid_epoch.append(a)       
    loss_valid_mean.append( np.average(loss_valid_epoch, axis=0, weights= weight_valid_epoch) )
    acc_valid_mean.append( np.mean( acc_valid_epoch, 0) )
    print '\t', ' '.join( [str(np.round(t,5)) for t in loss_valid_mean[-1][[0,2,3]]] ), acc_valid_mean[-1][0][4]

    if reduce_lr and ( epoch+1 == (num_epochs-start_epoch) * 0.5 or \
                      epoch+1 == (num_epochs-start_epoch) * 0.8 ):
        new_lr = sh_lr.get_value() * 0.1
        sh_lr.set_value(lasagne.utils.floatX(new_lr))
        
    ################   Dump Models and Log  ##########################

    with open((metadata_path + "-%d" % (epoch) + ".pkl"), 'w') as f:
        cPickle.dump({'config_name': config_name, 'param_values': lasagne.layers.get_all_param_values(l_out)}, f, \
                 protocol=pickle.HIGHEST_PROTOCOL)
       
    print >> log_file, epoch+1, 
    print >> log_file, ' '.join( [str(np.round(t,5)) for t in loss_train_mean[-1][[0,2,3]]] ), 
    print >> log_file, ' '.join( [str(np.round(t,5)) for t in loss_valid_mean[-1][[0,2,3]]]), 
    print >> log_file, ' '.join(map(str, np.round(acc_train_mean[-1],4))),
    print >> log_file, ' '.join(map(str, np.round(acc_valid_mean[-1][0],4))),
    print >> log_file, time.time() - start_time
log_file.close()
