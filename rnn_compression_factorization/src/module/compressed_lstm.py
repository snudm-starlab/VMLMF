################################################################################
# Starlab RNN-compression with factorization method : Lowrank and group-lowrank rnn
#
# Author: Hyojin Jeon (tarahjjeon@gmail.com), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# Version : 1.0
# Date : Nov 10, 2020
# Main Contact: Donghae Jang
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

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None,g=None, recurrent_init=None,
                 hidden_init=None,isShuffle=False):
        super(myLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
        self.g=g

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
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRanks]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRanks, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRanks, hidden_size]))
            self.U3 = nn.Parameter(0.1 * torch.randn([uRanks, hidden_size]))
            self.U4 = nn.Parameter(0.1 * torch.randn([uRanks, hidden_size]))

        self.bias_f = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_i = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_c = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_o = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, hiddenStates):
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
        return h_next, c_next

### 2. Group LSTM cell
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
        
        self.W_row=input_size if wRank is None else wRank
        self.W=None if wRank is None else nn.Parameter(0.1*torch.randn([self.input_size,self.wRank]))
        self.Ws=[]
        
        #case: only when uRank is not none
        #Ws=[W_f,W_i,W_c,W_o]
        for i in range(4): # of lstm cell gates
            self.Ws.append(nn.Parameter(0.1*torch.randn([self.W_row,hidden_size])))
        
        self.Ug=[[] for i in range(self.g)] # U, UU, UUU, UUUU ... > weight for group 1,2,3,4,...
        for idx,urank in enumerate(self.uRanks):
            self.Ug[idx].append(nn.Parameter(0.1*torch.randn([g,int(hidden_size/g),self.uRanks[idx]])))
            for i in range(4): #f,i,c,o
                self.Ug[idx].append(nn.Parameter(0.1*torch.randn([g,self.uRanks[idx],int(hidden_size/g)])))
        
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
        
    def forgetgate(self,x,h):
        wVal_f=torch.matmul(x,self.Ws[0]) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[0])
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))
        
        uVal_f=0
        
        for g_idx,groupWeight in enumerate(self.Ug):
            #groupWeight=[U,U_f,U_i,U_c,U_o]
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,groupWeight[0])
            uVal_f+=self.hiddenOperation(g_h,groupWeight[1],batch_size) if self.uRanks[g_idx] > 0 else 0
        
        return torch.sigmoid(wVal_f+uVal_f+self.bias_f)
    
    def inputgate(self,x,h):
        wVal_i=torch.matmul(x,self.Ws[1]) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[1])
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))
        
        uVal_i=0
        
        for g_idx,groupWeight in enumerate(self.Ug):
            #groupWeight=[U,U_f,U_i,U_c,U_o]
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,groupWeight[0])
            uVal_i+=self.hiddenOperation(g_h,groupWeight[2],batch_size) if self.uRanks[g_idx] > 0 else 0
        
        return torch.sigmoid(wVal_i+uVal_i+self.bias_i)
        
    def outgate(self,x,h):
        wVal_o=torch.matmul(x,self.Ws[2]) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[2])
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))

        uVal_o=0

        for g_idx,groupWeight in enumerate(self.Ug):
            #groupWeight=[U,U_f,U_i,U_c,U_o]
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,groupWeight[0])
            uVal_o+=self.hiddenOperation(g_h,groupWeight[3],batch_size) if self.uRanks[g_idx] > 0 else 0

        return torch.sigmoid(wVal_o+uVal_o+self.bias_o)   
        
    def gate_gate(self,x,h):
        wVal_g=torch.matmul(x,self.Ws[3]) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[3])
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))
        
        uVal_g=0
        
        for g_idx,groupWeight in enumerate(self.Ug):
            #groupWeight=[U,U_f,U_i,U_c,U_o]
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,groupWeight[0])
            uVal_g+=self.hiddenOperation(g_h,groupWeight[4],batch_size) if self.uRanks[g_idx] > 0 else 0
        
        return torch.tanh(wVal_g+uVal_g+self.bias_c)   
        
    def forward(self,x,hiddenStates):
        (h,c)=hiddenStates

        val_f=self.forgetgate(x,h)
        val_i=self.inputgate(x,h)
        val_o=self.outgate(x,h)s
        val_c=self.gate_gate(x,h)
        
        c_next=val_f*c+val_i*val_c
        h_next=val_o*torch.tanh(c_next)
        
        c_next=self.shuffle(c_next) if self.isShuffle else c_next
        h_next=self.shuffle(h_next) if self.isShuffle else h_next
        
        return h_next,c_next

### 3. LSTM network 
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
        print("cell:{}".format(self.cell))
        print("recurrent:{}\n".format(recurrent))
        print("start training with wRank:{}".format(wRank))
        print("start training with uRanks:{}".format(uRanks))

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
                h, c = cell(x_time[t], (h, c))
                outputs.append(h)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(h)
            i = i + 1

        return x, torch.cat(hiddens, -1)

