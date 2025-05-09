import sys
import os
import struct
import time
import h5py
from scipy.stats import pearsonr
from tqdm import tqdm
import math
from copy import copy
from sklearn.utils import resample

import numpy as np
import src_new.numpy_utility as pnu

import torch as T
import torch.nn as L
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim


def get_value(_x):
    return np.copy(_x.data.cpu().numpy())
def set_value(_x, x):
    _x.data.copy_(T.from_numpy(x))
    
    
def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual
        
def iterate_minibatches(inputs, targets, batchsize):
    '''return inputs.shape[0]//batchsize batches plus one residual batches smaller than batchsize if needed'''
    seq = ['|','/','--','\\']
    assert len(inputs) == len(targets)
    n = np.ceil(float(len(inputs)) / batchsize)
    for i,start_idx in enumerate(range(0, len(inputs), batchsize)):
        sys.stdout.write('\r%-2s: %.1f %%'%(seq[i%4], float(i+1)*100/n))
        excerpt = slice(start_idx, start_idx+batchsize)
        yield inputs[excerpt], targets[excerpt]  
    

################################################################  
def iterate_voxels(batch_params, voxel_params):
    seq = ['|','/','--','\\']
    batchsize = batch_params[0].size()[0]
    totalsize = voxel_params[0].shape[0]
    index = np.arange(batchsize)
    if batchsize==totalsize:
        for _p, p in zip(batch_params, voxel_params):
            set_value(_p, p)
        yield index
    else:
        n = np.ceil(float(totalsize)/batchsize)
        for i,startindex in enumerate(range(0, totalsize, batchsize)):    
            shifted_index = (index + startindex) % totalsize
            sys.stdout.write('\r%-2s: %.1f %%: voxels [%6d:%-6d] of %d' % (seq[i%4], float(i+1)*100/n, shifted_index[0], shifted_index[-1], totalsize))
            for _p, p in zip(batch_params, voxel_params):
                set_value(_p, p[shifted_index])
            yield shifted_index


################################################################
def iterate_slice(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield slice(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield slice(start+batch_count*batchsize,start+length),residual


######################################################
def iterate_subject_ordering_minibatches(inputs, targets, image_ordering, batchsize, shuffle=False):
    '''return inputs.shape[0]//batchsize batches plus one residual batches smaller than batchsize if needed'''
    batched_image_ordering = copy(image_ordering)
    batched_targets = np.arange(len(batched_image_ordering))
    if shuffle == True:
        shuffle_idx = resample(np.arange(len(batched_image_ordering)),
            replace=False)
        batched_image_ordering = batched_image_ordering[shuffle_idx]
        batched_targets = batched_targets[shuffle_idx]
    n_batches = int(np.floor(len(batched_image_ordering) / batchsize))
    max_samples = int(n_batches * batchsize)
    batched_image_ordering = batched_image_ordering[:max_samples]
    batched_targets = batched_targets[:max_samples]
    batched_image_ordering = np.reshape(batched_image_ordering, (n_batches,batchsize))
    batched_targets = np.reshape(batched_targets, (n_batches,batchsize))
    return batched_image_ordering, batched_targets


##################################################################
def subject_training_pass(_trn_fn, _ext, _cons, _ops, x, v, ordering, batch_size):
    trn_err = float(0)
    b_img, b_targ = iterate_subject_ordering_minibatches(x, v, ordering, batch_size, shuffle=True)
    for b in range(len(b_img)):
        trn_err += get_value(T.mean(_trn_fn(_ext, _cons, _ops, x[b_img[b]], v[b_targ[b]])))
    return trn_err / len(b_img)


#################################################
def subject_holdout_pass(_hld_fn, _ext, _cons, x, v, ordering, batch_size):
    val_err = float(0)
    b_img, b_targ = iterate_subject_ordering_minibatches(x, v, ordering, batch_size, shuffle=True)
    for b in range(len(b_img)):
        val_err += get_value(T.mean(_hld_fn(_ext, _cons, x[b_img[b]], v[b_targ[b]])))
    return val_err / len(b_img)


#################################################
def subject_pred_pass(_pred_fn, _ext, _con, x, batch_size):
    pred = _pred_fn(_ext, _con, x[:batch_size]) # this is just to get the shape
    pred = np.zeros(shape=(len(x), pred.shape[1]), dtype=np.float32) # allocate
    for rb,_ in iterate_range(0, len(x), batch_size):
        pred[rb] = get_value(_pred_fn(_ext, _con, x[rb]))
    return pred

def subject_validation_pass(_pred_fn, _ext, _con, x, v, ordering, batch_size):
    val_cc = np.zeros(shape=(v.shape[1]), dtype=v.dtype)
    val_pred = subject_pred_pass(_pred_fn, _ext, _con, x, batch_size)[ordering]
    for i in range(v.shape[1]):
        val_cc[i] = np.corrcoef(v[:,i], val_pred[:, i])[0,1]
    return val_cc

#################################################
def random_split(stim, voxel, subselect, trn_size, holdout_frac, random=False):
    n = len(stim)
    holdout_size = int(np.ceil(n * holdout_frac))
    if random:
        idx = np.arange(n)
        np.random.shuffle(idx)  
        idx = idx[:trn_size]
        return stim[idx[:-holdout_size]], voxel[:,subselect][idx[:-holdout_size]], \
               stim[idx[-holdout_size:]], voxel[:,subselect][idx[-holdout_size:]]
    else:
        return stim[:trn_size-holdout_size], voxel[:,subselect][:trn_size-holdout_size], \
               stim[-holdout_size:], voxel[:,subselect][-holdout_size:]

def learn_params_(writer, _trn_fn, _hld_fn, _pred_fn, _ext, _cons, _opts, stims,
    trn_voxels, trn_stim_ordering, hld_voxels, hld_stim_ordering, num_epochs,
    batch_size, holdout_frac=0.1, trn_size=None, masks=None):
    '''assumes shared_model and subject_fwrfs in global scope
    
       voxelwise model fit is performed for one subject at a time.
    
    '''
    import copy

    hold_hist, trn_hist = [], []
    best_joint_cc_score = float(0)
    best_params = {}
    best_epoch = 0

    for epoch in range(num_epochs):
        ##
        ## Training pass for this subject
        ##
        start_time = time.time()
        _ext.train()
        _cons.train()
        trn_err = subject_training_pass(_trn_fn, _ext, _cons, _opts, stims,
            trn_voxels, trn_stim_ordering, batch_size)
        trn_hist += [trn_err,]
        ##
        _ext.eval()
        _cons.eval()
        hold_err = subject_holdout_pass(_hld_fn, _ext, _cons, stims,
            hld_voxels, hld_stim_ordering, batch_size)
        hold_hist += [hold_err,]
        ##
        ## Do a validation pass to monitor the evolution
        ##
        print("\n  Epoch {} of {} took       {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:               {:.6f}".format(trn_err))
        print("  holdout loss (batch):        {:.6f}".format(hold_err))
        ##
        joint_cc_score = np.nan_to_num(subject_validation_pass(_pred_fn,
            _ext, _cons, stims, hld_voxels, hld_stim_ordering, batch_size))
        joint_cc_score = np.mean(joint_cc_score)
        ##
        ## Save parameter snapshot
        ##
        if joint_cc_score>best_joint_cc_score:
            best_joint_cc_score = joint_cc_score
            best_epoch = epoch
            best_params = {
                'enc': copy.deepcopy(_ext.state_dict()),
                'fwrf': copy.deepcopy(_cons.state_dict())
                }
        sys.stdout.flush()
        writer.add_scalars('Loss', {'train': trn_err, 'val': hold_err}, epoch+1)
        writer.flush()
        writer.add_scalars('Acc', {'val': joint_cc_score}, epoch+1)
        writer.flush()
    ###
    writer.close()
    final_params = {
        'enc': copy.deepcopy(_ext.state_dict()),
        'fwrf': copy.deepcopy(_cons.state_dict())
        }

    return best_params, final_params, hold_hist, trn_hist, best_epoch, best_joint_cc_score



def validation_(_pred_fn, _ext, _cons, stims, voxels, ordering, batch_size, masks=None):
    val_ccs = {}
    _ext.eval()
    for s,v in voxels.items():
        if masks is None:
            mask = np.ones(shape=(v.shape[1]), dtype=bool) 
        else:
            mask = masks[s]
        _ext.eval()
        _cons[s].eval()   
        val_ccs[s] = np.nan_to_num(subject_validation_pass(_pred_fn, _ext, _cons[s], stims[s], v[:,mask], ordering[s], batch_size))
    return val_ccs


