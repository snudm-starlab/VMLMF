################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
# Project: Starlab 
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: compression_cal.py
# - utilities for analyze compression results of VMLMF 
#
# Version : 1.0
# Date : Oct 14, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
import sys
import torch
from torch.autograd import Variable
sys.path.append('../')

def print_model_parm_nums(model):
    """
    print the number of parameters of the model
    @param model
    model to count parameters
    """
    modelparams=sum(p.numel() for p in model.parameters())
    print(" + Number of params:{:.2f}K".format(modelparams/1e3))

def print_model_parm_flops(model,seq_len,args,modeltype="vmmodel"): #model:Net, maxsteps:args.max_steps, batch_size:args.batch_size
    """
    print FLOPs of the model
    @param model
    model to count FLOPs
    @param seq_len
    sequence length of the input data
    @param args
    argument user decided
    @param modeltype
    type of the model
    
    * vmlmf_group, vmlmf_lm is not implemented
    """
    if modeltype in ['vmlmf_group','vmlmf_lm']:
        print("Not Implemented")
        return
    batch_size=args.batch_size
    modeltype=args.model.lower() if modeltype != "mylstm" else "mylstm"  #vmlmf have to more compute FLOPs of VM module
    total_ops=count_lstm(model,seq_len,batch_size,modeltype)
    total_ops+=count_linear(model,18) #linear의 iN_FEATURES,OUTFEATURE
    print("  + Number of FLOPs: {:.2f}M".format(total_ops / 1e6))
    print(total_ops)

def print_model_parm_names(model):
    """
    print the name of parameters of the model
    @param model
    model to get parameters
    """
    
    for idx,m in enumerate(model.modules()):
        print( idx, '->', m )

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def _count_lstm_cell(modeltype,input_size,hidden_size,wRank=None,uRank=None,bias=True):
    """
    count FLOPs of lstm/vmlmf cell
    @param modeltype
    modeltype
    @param input_size
    input size of the model
    @param hidden_size
    hidden layer size of the model
    @param wRank, uRank
    rank of vmlmf
    @ param bias
    whether the model share bias between for gates
    
    * vmlmf_group, vmlmf_lm is not implemented
    """
    total_ops=0
    isvmmodel = True if modeltype != "mylstm" else False
    #vector-vector multiplication
    input_dia_ops  = input_size 
    hidden_dia_ops = hidden_size 
    #substract vec elem
    if wRank is not None:
        input_addition = (2*wRank-1)*input_size + hidden_size
    if uRank is not None:
        hidden_addition = (2*uRank-1)*hidden_size +hidden_size

    input_ops=(2*input_size-1)*wRank+(2*wRank-1)*hidden_size if isvmmodel else (2*input_size-1)*hidden_size
    hidden_ops=(2*hidden_size-1)*uRank+(2*uRank-1)*hidden_size if isvmmodel else (2*hidden_size-1)*hidden_size
    state_ops=input_ops+hidden_ops + input_dia_ops + hidden_dia_ops +hidden_size*3 +input_addition + hidden_addition if isvmmodel else input_ops + hidden_ops + hidden_size
    
    if bias:
        state_ops+=hidden_size
    total_ops+=state_ops*4

    #hadamard addition (f*c + i*g )
    total_ops+=hidden_size*3

    #h'=o*tanh(c')
    total_ops+=hidden_size

    return total_ops

def count_lstm(model,seq_len,batch_size,modeltype):
    """
    count FLOPs of lstm/vmlmf layer 
    @param model
    model object
    @param seq_len
    sequence length of the input data
    @param batch_size
    batch_size of the input data
    @param modeltype
    type of the model
    
    * vmlmf_group, vmlmf_lm is not implemented
    """
    if modeltype in ['vmlmf_group','vmlmf_lm']:
        print("Not Implemented")
        return
    total_ops=0
    total_ops+=_count_lstm_cell(modeltype,model.rnn.input_size,model.rnn.hidden_layer_sizes[0],model.rnn.wRank,model.rnn.uRanks,bias=True) 
    for i in range(len(model.rnn.hidden_layer_sizes)-1):
        total_ops+=_count_lstm_cell(modeltype,model.rnn.hidden_layer_sizes[i],model.rnn.hidden_layer_sizes[i+1],model.rnn.wRank,model.rnn.uRanks,bias=True)
    total_ops*=seq_len
    total_ops*=batch_size

    return total_ops 
def count_linear(model,output_size): # 10*64 + 10 (10:model&linear layer output size,64:model batch size&linear_layer input_size /bias)
    """
    count FLOPs of linear layer
    @param model
    model object
    @param output_size
    output size of the model
    
    * vmlmf_group, vmlmf_lm is not implemented
    """
    input_size=model.rnn.hidden_layer_sizes[-1]
    return input_size*output_size*2

    
