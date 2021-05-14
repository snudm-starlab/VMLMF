################################################################################
# Starlab RNN-compression with factorization method : Lowrank and group-lowrank rnn
#
# Author: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# Version : 1.0
# Date : Nov 10, 2020
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
import torch
from torch.nn import Parameter, ParameterList
import torch.nn as nn
import torch.nn.functional as F
import math

#MV_LMF LSTM cell
class myDualDiagonalLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, recurrent_init=None,isMask=False,isBand=True,
                 hidden_init=None,g=3,isShuffle=False,isdiagonal=True, initdiagonal=0.1111,zerodiagonal=False): #if isdiagonal: unblock diagonal else block diagonal // if zerodiagonal: change diagonal to zero when LMF
        super(myDualDiagonalLSTMCell, self).__init__()
        print("init diagonal cell __ diagonal elementwise multiplication and lowrank factorization with off diagonal elements")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks 
       
        self.cnt=0
        
        if self.wRank is not None:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
        if self.uRanks is not None:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRanks]))

        wrow=self.input_size if self.wRank is None else self.wRank
        urow=self.hidden_size if self.uRanks is None else self.uRanks

        self.Ws=nn.ParameterList([nn.Parameter(0.1*torch.randn([wrow,hidden_size])) for _ in ["f","i","c","o"]])
        self.Us=nn.ParameterList([nn.Parameter(0.1 * torch.randn([urow, hidden_size])) for _ in ["f","i","c","o"]])
        self.biases=nn.ParameterList([nn.Parameter(torch.ones([1,hidden_size])) for _ in ["f","i","c","o"]])


    def forward(self, x, hiddenStates,device):
        #step 01. diagonal elements vetor * x vector element-wise multiplication
        #step 02. offdiagonal elements low rank approximation * x vector multiplication
        #step 03. add 2 vectors from previous process
        (h, c) = hiddenStates
        batch_size=h.shape[0]
        
        paddingTensor=torch.zeros([batch_size,self.hidden_size-self.input_size]).to(device) if self.hidden_size-self.input_size else None
     
        output_of={}
        for idx,gate in enumerate(['forget','input','gate','output']):
            
            W=self.Ws[idx] if self.wRank is None else torch.matmul(self.W, self.Ws[idx])
            U=self.Us[idx] if self.uRanks is None else torch.matmul(self.U, self.Us[idx])

            #input calculation
            input_dia_mul=torch.cat([torch.diagonal(W,0)*x.squeeze(),paddingTensor],dim=1) if paddingTensor is not None else torch.diagonal(W,0)*x.squeeze()
            input_lmf_mul=torch.matmul(x,W) 

            #hidden state calculation
            hstate_dia_mul=torch.diagonal(U,0)*h.squeeze()
            hstate_lmf_mul=torch.matmul(h, U)

            unactivated_output=input_dia_mul+input_lmf_mul+hstate_dia_mul+hstate_lmf_mul+self.biases[idx]
            activated_output=torch.sigmoid(unactivated_output) if gate is not 'gate' else torch.tanh(unactivated_output)
            
            output_of[gate]=activated_output

        c_next = output_of['forget'] * c + output_of['input'] * output_of['gate']
        h_next = output_of['output']* torch.tanh(c_next)
        
        self.cnt+=1  
        return h_next, c_next


### 0. Diagonal LSTM Cell
class myDiagonalLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, recurrent_init=None,
                 hidden_init=None,g=3,isShuffle=False,isdiagonal=True):
        super(myDiagonalLSTMCell, self).__init__()
        print("init diagonal cell __ use only diagonal elements")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks 
        if type(self.uRanks) is list:
            self.uRanks=uRanks[0]
        self.g=g
        self.cnt=0
        unit_block=torch.zeros(int(self.hidden_size/self.g),int(self.hidden_size/self.g)).fill_(0.1321)
        self.blockdiagonal_h_weight=torch.block_diag(unit_block,unit_block,unit_block) #diagonal block 3
        self.diagonal_i_weight=torch.zeros(self.input_size,self.hidden_size).fill_diagonal_(0.1321) if isdiagonal else self.blockdiagonal_h_weight
        self.diagonal_h_weight=torch.zeros(self.hidden_size,self.hidden_size).fill_diagonal_(0.1231) if isdiagonal else self.blockdiagonal_h_weight

        self.W1 = nn.Parameter(self.diagonal_i_weight)
        self.W2 = nn.Parameter(self.diagonal_i_weight)
        self.W3 = nn.Parameter(self.diagonal_i_weight)
        self.W4 = nn.Parameter(self.diagonal_i_weight)
        
        
        self.U1 = nn.Parameter(self.diagonal_h_weight)
        self.U2 = nn.Parameter(self.diagonal_h_weight)
        self.U3 = nn.Parameter(self.diagonal_h_weight)
        self.U4 = nn.Parameter(self.diagonal_h_weight)
     
        self.bias_f = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_i = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_c = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_o = nn.Parameter(torch.ones([1, hidden_size]))

 
    def forward(self, x, hiddenStates,device):
        (h, c) = hiddenStates
        wVal1 = x.squeeze()*torch.diagonal(self.W1,0)
        wVal2 = x.squeeze()*torch.diagonal(self.W2,0)
        wVal3 = x.squeeze()*torch.diagonal(self.W3,0)
        wVal4 = x.squeeze()*torch.diagonal(self.W4,0)
       
        uVal1 = x.squeeze()*torch.diagonal(self.U1,0)
        uVal2 = x.squeeze()*torch.diagonal(self.U2,0)
        uVal3 = x.squeeze()*torch.diagonal(self.U3,0)
        uVal4 = x.squeeze()*torch.diagonal(self.U4,0)
       
        matVal_i = wVal1 + uVal1
        matVal_f = wVal2 + uVal2
        matVal_o = wVal3 + uVal3
        matVal_c = wVal4 + uVal4

        i = torch.sigmoid(matVal_i + self.bias_i)
        f = torch.sigmoid(matVal_f + self.bias_f)
        o = torch.sigmoid(matVal_o + self.bias_o)

        c_tilda = torch.tanh(matVal_c + self.bias_c)

        c_next = f * c + i * c_tilda
        h_next = o * torch.tanh(c_next)
        if self.cnt % 10000==0:
            print(self.W1)
            print("recurrent weights")
            print(self.U1)
        self.cnt+=1   
        return h_next, c_next

### 1. Vanilla LSTM Cell (no group)
class myLSTMCell(nn.Module):
    '''
    LR - Low Rank
    LSTM LR Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_nonlinearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of all W matrices
    (creates 5 matrices if not None else creates 4 matrices)
    uRank = rank of all U matrices
    (creates 5 matrices if not None else creates 4 matrices)

    Basic architecture:

    f_t = gate_nl(W1x_t + U1h_{t-1} + B_f)
    i_t = gate_nl(W2x_t + U2h_{t-1} + B_i)
    C_t^ = update_nl(W3x_t + U3h_{t-1} + B_c)
    o_t = gate_nl(W4x_t + U4h_{t-1} + B_o)
    C_t = f_t*C_{t-1} + i_t*C_t^
    h_t = o_t*update_nl(C_t)

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    '''

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, recurrent_init=None,
                 hidden_init=None,g=1,isShuffle=False):
        super(myLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks 
        if type(self.uRanks) is list:
            self.uRanks=uRanks[0]
        self.g=g
        self.cnt=0
        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W4 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W4 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRanks is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U4 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            print("uRanks {} type is {}".format(uRanks,type(uRanks)))
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRanks]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRanks, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRanks, hidden_size]))
            self.U3 = nn.Parameter(0.1 * torch.randn([uRanks, hidden_size]))
            self.U4 = nn.Parameter(0.1 * torch.randn([uRanks, hidden_size]))

        self.bias_f = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_i = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_c = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_o = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, hiddenStates,device):
        (h, c) = hiddenStates

        if self.wRank is None:
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
            wVal4 = torch.matmul(x, self.W4)
        else:
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)
            wVal4 = torch.matmul(
                torch.matmul(x, self.W), self.W4)

        if self.uRanks is None:
            uVal1 = torch.matmul(h, self.U1)
            uVal2 = torch.matmul(h, self.U2)
            uVal3 = torch.matmul(h, self.U3)
            uVal4 = torch.matmul(h, self.U4)
        else:
            uVal1 = torch.matmul(
                torch.matmul(h, self.U), self.U1)
            uVal2 = torch.matmul(
                torch.matmul(h, self.U), self.U2)
            uVal3 = torch.matmul(
                torch.matmul(h, self.U), self.U3)
            uVal4 = torch.matmul(
                torch.matmul(h, self.U), self.U4)
        matVal_i = wVal1 + uVal1
        matVal_f = wVal2 + uVal2
        matVal_o = wVal3 + uVal3
        matVal_c = wVal4 + uVal4

        i = torch.sigmoid(matVal_i + self.bias_i)
        f = torch.sigmoid(matVal_f + self.bias_f)
        o = torch.sigmoid(matVal_o + self.bias_o)

        c_tilda = torch.tanh(matVal_c + self.bias_c)

        c_next = f * c + i * c_tilda
        h_next = o * torch.tanh(c_next)
        self.cnt+=1

        if self.cnt%100000==0:
            print(self.U1)
        return h_next, c_next
### 3. Group LSTM cell
class myLSTMGroupCell(nn.Module):
  
    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None,g=2,recurrent_init=None,
                 hidden_init=None,isShuffle=False):
        super(myLSTMGroupCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
        self.g=g
        self.isShuffle=isShuffle
        W_row=input_size if wRank is None else wRank
        self.W=None if wRank is None else nn.Parameter(0.1*torch.randn([input_size,wRank]))
        
        #case: only when uRank is not none
        #Ws=[W_f,W_i,W_c,W_o]
        self.Ws=nn.ParameterList([nn.Parameter(0.1*torch.randn([W_row,hidden_size])) for _ in ["f","i","c","o"]])
        self.Us=nn.ParameterList([nn.Parameter(0.1*torch.randn([g,int(hidden_size/g),urank])) for urank in uRanks]) # U, UU, UUU, UUUU ... > weight for group 1,2,3,4,...
        self.Ugate=nn.ParameterList()
        for g_idx in range(g):
            self.Ugate.extend(nn.ParameterList([nn.Parameter(0.1*torch.randn([g,uRanks[g_idx],int(hidden_size/g)])) for _ in ["f","i","c","o"]]))
        '''
        for idx,urank in enumerate(uRanks):
            self.Ug[idx].append()
            for i in range(4): #f,i,c,o
                self.Ug[idx].append(nn.Parameter(0.1*torch.randn([g,uRanks[idx],int(hidden_size/g)])))
        '''
        self.bias_f=nn.Parameter(torch.ones([1,hidden_size]))
        self.bias_i=nn.Parameter(torch.ones([1,hidden_size]))
        self.bias_c=nn.Parameter(torch.ones([1,hidden_size]))
        self.bias_o=nn.Parameter(torch.ones([1,hidden_size]))
            
    def hiddenOperation(self,g_h,groupWeight,batch_size):
        #g_h transposed and multiplied with U, groupWeight=gate's weight for each group
        g_h=torch.bmm(g_h,groupWeight)
        g_h=torch.transpose(g_h,0,1)
        g_h=g_h.contiguous().view(batch_size,self.hidden_size)
        return g_h
        
    def forgetgate(self,x,h,device):
        wVal_f=torch.matmul(x,self.Ws[0].to(device)) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[0].to(device))
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))
        
        uVal_f=0
        
        for g_idx in range(self.g):
            g_gate=g_idx*4 #4:f,i,c,o
            # Us=[] group 별 U
            # groupWeight=[U_f,U_i,U_c,U_o]
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,self.Us[g_idx].to(device))
            uVal_f+=self.hiddenOperation(g_h,self.Ugate[g_gate].to(device),batch_size) if self.uRanks[g_idx] > 0 else 0
        
        return torch.sigmoid(wVal_f+uVal_f+self.bias_f)
    
    def inputgate(self,x,h,device):
        wVal_i=torch.matmul(x,self.Ws[1].to(device)) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[1].to(device))
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))
        
        uVal_i=0
        
        for g_idx in range(self.g):
            g_gate=g_idx*4 #4:f,i,c,o
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,self.Us[g_idx].to(device))
            uVal_i+=self.hiddenOperation(g_h,self.Ugate[g_gate+1].to(device),batch_size) if self.uRanks[g_idx] > 0 else 0
        
        return torch.sigmoid(wVal_i+uVal_i+self.bias_i)
        
    def outgate(self,x,h,device):
        wVal_o=torch.matmul(x,self.Ws[2].to(device)) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[2].to(device))
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))

        uVal_o=0

        for g_idx in range(self.g):
            #groupWeight=[U_f,U_i,U_c,U_o]
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_gate=g_idx*4 #4:f,i,c,o
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,self.Us[g_idx].to(device))
            uVal_o+=self.hiddenOperation(g_h,self.Ugate[g_gate+2].to(device),batch_size) if self.uRanks[g_idx] > 0 else 0

        return torch.sigmoid(wVal_o+uVal_o+self.bias_o)   
        
    def gate_gate(self,x,h,device):
        wVal_g=torch.matmul(x,self.Ws[3].to(device)) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[3].to(device))
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))
        
        uVal_g=0
        
        for g_idx in range(self.g):
            #groupWeight=[U,U_f,U_i,U_c,U_o]
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_gate=g_idx*4 #4:f,i,c,o
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,self.Us[g_idx].to(device))
            uVal_g+=self.hiddenOperation(g_h,self.Ugate[g_gate+3].to(device),batch_size) if self.uRanks[g_idx] > 0 else 0
        
        return torch.tanh(wVal_g+uVal_g+self.bias_c)   

    def shuffle(self,x):
        perm=torch.randperm(x.shape[-1])
        return x[:,perm]   

    def forward(self,x,hiddenStates,device):
        (h,c)=hiddenStates

        val_f=self.forgetgate(x,h,device)
        val_i=self.inputgate(x,h,device)
        val_o=self.outgate(x,h,device)
        val_c=self.gate_gate(x,h,device)
        
        c_next=val_f*c+val_i*val_c
        h_next=val_o*torch.tanh(c_next)
        
        c_next=self.shuffle(c_next) if self.isShuffle else c_next
        h_next=self.shuffle(h_next) if self.isShuffle else h_next
        
        return h_next,c_next

class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRanks=None, recurrent=False,cell=myLSTMCell,g=None,isShuffle=False, **kwargs):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRanks = uRanks
        self.isShuffle=isShuffle
        self.g=g
        self.cell=cell
        if g is None or g <2:
            self.uRanks=uRanks[0] if type(uRanks) is list else uRanks
        print("#1.cell:{}".format(self.cell))
        print("#2.recurrent:{}".format(recurrent))
        print("#start training with wRank:{}".format(wRank))
        print("#start training with uRanks:{}".format(uRanks))

        
        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1

        rnn_cells = []
        in_size = input_size
        i = 0
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if recurrent_inits is not None:
                kwargs["recurrent_init"] = recurrent_inits[i]
            if hidden_inits is not None:
                kwargs["hidden_init"] = hidden_inits[i]
            rnn_cells.append(self.cell(in_size, hidden_size, wRank=self.wRank, uRanks=self.uRanks,isShuffle=self.isShuffle,g=self.g, **kwargs))
            in_size = hidden_size

        self.rnncells = nn.ModuleList(rnn_cells)

        # h0 = torch.zeros(hidden_size * num_directions, requires_grad=False)
        # self.register_buffer('h0', h0)

    def forward(self, x, hidden=None):
        time_index = self.time_index
        batch_index = self.batch_index
        hiddens = []

        i = 0
        for cell in self.rnncells:
            # hx = self.h0.unsqueeze(0).expand(
            #    x.size(batch_index),
            #    self.hidden_size * num_directions).contiguous()
            self.device = x.device
            h = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            c = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                h, c = cell(x_time[t], (h, c),self.device)
                outputs.append(h)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(h)
            i = i + 1

        return x, torch.cat(hiddens, -1)

        return x, torch.cat(hiddens, -1)

