import sys
import torch
from torch.autograd import Variable
sys.path.append('../')

def print_model_parm_nums(model):
    modelparams=sum(p.numel() for p in model.parameters())
    print(" + Number of params:{:.2f}K".format(modelparams/1e3))

def print_model_parm_flops(model,seq_len,batch_size): 
    total_ops=count_lstm(model,seq_len,batch_size)
    total_ops+=count_linear(model,10) #10: # of OUTFEATURE in linear layer
    print("  + Number of FLOPs: {:.2f}M".format(total_ops / 1e6))
    print(total_ops)

def print_model_parm_names(model):
    
    for idx,m in enumerate(model.modules()):
        print( idx, '->', m )

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def _count_lstm_cell(input_size,hidden_size,wRank=None,uRank=None,bias=True):
    
    total_ops=0

    #vector-vector multiplication
    input_dia_ops  = input_size+hidden_size
    hidden_dia_ops = hidden_size + hidden_size

    input_ops=input_size*hidden_size if wRank is None else (input_size+hidden_size)*wRank
    hidden_ops=hidden_size*hidden_size if uRank is None else (hidden_size*2)*uRank
    state_ops=input_ops+hidden_ops+hidden_size + input_dia_ops + hidden_dia_ops
    
    if bias:
        state_ops+=hidden_size
    total_ops+=state_ops*4

    #hadamard addition (f*c + i*g )
    total_ops+=hidden_size*3

    #h'=o*tanh(c')
    total_ops+=hidden_size

    return total_ops

def count_lstm(model,seq_len,batch_size):
    
    total_ops=0
    total_ops+=_count_lstm_cell(model.rnn.input_size,model.rnn.hidden_layer_sizes[0],model.rnn.wRank,model.rnn.uRanks,bias=True)
    for i in range(len(model.rnn.hidden_layer_sizes)-1):
        total_ops+=_count_lstm_cell(model.rnn.input_size,model.rnn.hidden_layer_sizes[0],model.rnn.wRank,model.rnn.uRanks,bias=True)
    total_ops*=seq_len
    total_ops*=batch_size

    return total_ops 
def count_linear(model,output_size): # 10*64 + 10 (10:model&linear layer output size,64:model batch size&linear_layer input_size /bias)
    input_size=model.rnn.hidden_layer_sizes[-1]
    return input_size*output_size+output_size

    
