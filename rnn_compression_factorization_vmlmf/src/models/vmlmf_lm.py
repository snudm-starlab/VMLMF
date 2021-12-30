################################################################################
# [VMLMF] Low_rank Matrix Factorization with Vector-Multiplication
# Project: Starlab
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: vmlmf_lm.py
# - Cell and network class for VMLMF - language modeling
#
# Version : 1.0
# Date : Oct 14, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
# reference: https://github.com/ahmetumutdurmus/zaremba
################################################################################
# pylint: disable=C0103, E1101, C0114, R0902,C0116, R0914, R0913, C0123, W0613, W0102, R0201
"""
====================================
 :mod:`vmlmf_lm`
====================================
.. moduleauthor:: Hyojin Jeon  <tarahjjeon@snu.ac.kr>
설명
=====
VMLMF 기반 언어 모델 생성을 위한 모듈입니다.

"""
import torch
from torch import nn

class Embed(nn.Module):
    """Embedding layer

        :param int vocab_size: size of vocabulary
        :param int embed_size: size of embedding layer
    """
    def __init__(self, vocab_size, embed_size):
        """Initialize Embedding layer"""
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, embed_size))

    def forward(self, x):
        return self.W[x]

    def __repr__(self):
        return f"Embedding(vocab: {self.vocab_size}, embedding: {self.embed_size})"

class myVMLSTM_Group(nn.Module):
    """VMLSTM Group layer for language model

        :param int input_size: size of input vector
        :param int hidden_size: size of hidden state vector
        :param float dropout: drop out rate
        :param float winit: the bound of the uniform distribution
        :param int w_rank: rank of all input to hidden matrices
        :param int u_ranks: rank of all hidden to hidden matrices
        :param int g: number of groups
    """

    def __init__(self, input_size, hidden_size, dropout = 0, winit = 0.1,\
        w_rank=None,u_ranks=None,g=2):
        """Initialize VMLSTM Group layer"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.g=g

        self.w_rank=w_rank
        self.u_ranks=u_ranks

        #In-Hid in LMF
        self.u_x=nn.Parameter(torch.Tensor(input_size,w_rank))
        self.w_x = nn.Parameter(torch.Tensor(4 * hidden_size, w_rank))
        #Hid-Hid in group_LMF
        self.u_h = nn.ParameterList([nn.Parameter(torch.Tensor(
            g,int(hidden_size/g),u_ranks[g_idx])) for g_idx in range(self.g)])
        self.V_h = nn.ParameterList([ nn.Parameter(torch.Tensor(
            g,u_ranks[g_idx],4 * int(hidden_size/g))) for g_idx in range(self.g)])

        #bias
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))

        #diagonal vector
        self.dia_x=nn.Parameter(torch.Tensor(1,input_size))
        self.dia_h=nn.Parameter(torch.Tensor(1,hidden_size))
        self.cnt=0

    def __repr__(self):
        return f"LSTM(input: {self.input_size}, hidden: {self.hidden_size})"

    def lstm_step(self, x, h, c, w_x, V_h, b_x, b_h):
        """Forward computation of VMLSTM_Group Cell

        :param tensor x: input sequence
        :param tensor h: hidden state
        :param tensor c: cell state vector
        :param tensor w_x: input to hidden weights
        :param tensor w_h: hidden to hidden weights
        :param tensor b_x: input to hidden bias
        :param tensor b_h: hidden to hidden bias
        :returns: next hidden, cell states
        """
        dev = next(self.parameters()).device

        #save vm - redundant values
        vm_refined_x=torch.zeros(40,4*self.hidden_size,device=dev)
        vm_refined_h=torch.zeros(40,4*self.hidden_size,device=dev)

        #vm (for all 4 gates)
        vm_x=self.dia_x*x.squeeze()
        vm_h=self.dia_h*h.squeeze()
        vm_x=torch.cat([vm_x for _ in range(4)],dim=1)
        vm_h=torch.cat([vm_h for _ in range(4)],dim=1)

        lowered_x=torch.matmul(torch.matmul(x,self.u_x),self.w_x.t())

        index=list(range(self.g))
        for partial_h in range(self.g):
            h_top=h.view(-1,self.g,int(self.hidden_size/self.g))
            index=index[1:]+index[0:1] if partial_h>0 else index
            h_top=h_top[:,index,:] if partial_h > 0 else h_top
            h_top=torch.transpose(h_top,0,1)

            U_dia=self.u_h[partial_h]
            V_dia=self.V_h[partial_h]
            h_top=torch.bmm(h_top,U_dia)
            h_top=torch.bmm(h_top,V_dia)
            h_top=torch.transpose(h_top,0,1)
            h_top=h_top.contiguous().view(-1,self.hidden_size*4)
            lowered_h=h_top if partial_h==0 else h_top + lowered_h

        #cal redundant values
        hidden_size=self.hidden_size
        input_size=self.input_size
        re_Uh=self.u_h[0].view(self.hidden_size,self.u_ranks[0])
        re_Vh=torch.transpose(self.V_h[0],1,2).contiguous().view(4*self.hidden_size,self.u_ranks[0])

        for gate_idx in range(0,4*hidden_size,hidden_size):
            vm_refined_x[:,gate_idx:gate_idx+input_size]=x*torch.sum(\
                (self.u_x*self.w_x[gate_idx:gate_idx+input_size,:]),dim=1)
            vm_refined_h[:,gate_idx:gate_idx+hidden_size]=h*torch.sum(\
                (re_Uh*re_Vh[gate_idx:gate_idx+hidden_size,:]),dim=1)
        gx=vm_x+lowered_x-vm_refined_x+self.b_x
        gh=vm_h+lowered_h-vm_refined_h+self.b_h


        #total
        xi, xf, xo, xn = gx.squeeze().chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)

        inputgate = torch.sigmoid(xi + hi)
        forgetgate = torch.sigmoid(xf + hf)
        outputgate = torch.sigmoid(xo + ho)
        newgate = torch.tanh(xn + hn)
        c = forgetgate * c + inputgate * newgate
        h = outputgate * torch.tanh(c)
        return h, c

    #Takes input tensor x with dimensions: [T, B, X]
    def forward(self, x, states):
        h, c = states
        outputs = []
        inputs = x.unbind(0)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.w_x, self.V_h, self.b_x, self.b_h)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

######## <End of custom vmlmf lstm model code > ############

class myVMLSTM(nn.Module):
    """VMLSTM layer for language model

        :param int input_size: size of input vector
        :param int hidden_size: size of hidden state vector
        :param float dropout: drop out rate
        :param float winit: the bound of the uniform distribution
        :param int w_rank: rank of all input to hidden matrices
        :param int u_ranks: rank of all hidden to hidden matrices
    """
    def __init__(self, input_size, hidden_size, dropout = 0, winit = 0.1,\
        w_rank=None,u_ranks=None,device=None):
        """Initialize VMLSTM layer"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.w_rank=w_rank
        self.u_ranks=u_ranks

        #U in LMF
        self.u_x=nn.Parameter(torch.Tensor(input_size,w_rank))
        self.u_h=nn.Parameter(torch.Tensor(hidden_size,u_ranks))

        #V in LMF
        self.w_x = nn.Parameter(torch.Tensor(4 * hidden_size, w_rank))
        self.w_h = nn.Parameter(torch.Tensor(4 * hidden_size, u_ranks))

        #bias
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))

        #diagonal vector
        self.dia_x=nn.Parameter(torch.Tensor(1,input_size))
        self.dia_h=nn.Parameter(torch.Tensor(1,hidden_size))

        #save navigator
        self.cnt=0


    def __repr__(self):
        return f"LSTM(input: {self.input_size}, hidden: {self.hidden_size})"

    def lstm_step(self, x, h, c, w_x, w_h, b_x, b_h):
        """Forward computation of VMLSTM Cell

        :param tensor x: input sequence
        :param tensor h: hidden state
        :param tensor c: cell state vector
        :param tensor w_x: input to hidden weights
        :param tensor w_h: hidden to hidden weights
        :param tensor b_x: input to hidden bias
        :param tensor b_h: hidden to hidden bias
        :returns: next hidden, cell states
        """
        dev = next(self.parameters()).device

        #save vm - redundant values
        vm_refined_x=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)
        vm_refined_h=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)

        #vm (for all 4 gates)
        vm_x=self.dia_x*x.squeeze()
        vm_h=self.dia_h*h.squeeze()
        vm_x=torch.cat([vm_x for _ in range(4)],dim=1)
        vm_h=torch.cat([vm_h for _ in range(4)],dim=1)

        lowered_x=torch.matmul(torch.matmul(x,self.u_x),self.w_x.t())
        lowered_h=torch.matmul(torch.matmul(h,self.u_h),self.w_h.t())
        #cal redundant values
        hidden_size=self.hidden_size
        input_size=self.input_size
        for gate_idx in range(0,4*hidden_size,hidden_size):
            vm_refined_x[:,gate_idx:gate_idx+input_size]=x*torch.sum(\
                (self.u_x*self.w_x[gate_idx:gate_idx+input_size,:]),dim=1)
            vm_refined_h[:,gate_idx:gate_idx+hidden_size]=h*torch.sum(\
                (self.u_h*self.w_h[gate_idx:gate_idx+hidden_size,:]),dim=1)
        gx=vm_x+lowered_x-vm_refined_x+self.b_x
        gh=vm_h+lowered_h-vm_refined_h+self.b_h

        #total
        xi, xf, xo, xn = gx.chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)

        inputgate = torch.sigmoid(xi + hi)
        forgetgate = torch.sigmoid(xf + hf)
        outputgate = torch.sigmoid(xo + ho)
        newgate = torch.tanh(xn + hn)
        c = forgetgate * c + inputgate * newgate
        h = outputgate * torch.tanh(c)
        return h, c

    #Takes input tensor x with dimensions: [T, B, X]
    def forward(self, x, states):
        h, c = states
        outputs = []
        inputs = x.unbind(0)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.w_x, self.w_h, self.b_x, self.b_h)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

#My custom written LSTM module.
class LSTM(nn.Module):
    """LSTM layer for language model

    :param int input_size: size of input vector
    :param int hidden_size: size of hidden state vector
    :param float dropout: drop out rate
    :param float winit: the bound of the uniform distribution
    """

    def __init__(self, input_size, hidden_size, dropout = 0, winit = 0.1):
        """Initialize LSTM layer"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.w_x = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.w_h = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))

    def __repr__(self):
        return f"LSTM(input: {self.input_size}, hidden: {self.hidden_size})"

    def lstm_step(self, x, h, c, w_x, w_h, b_x, b_h):
        """Forward computation of LSTM Cell

        :param tensor x: input sequence
        :param tensor h: hidden state
        :param tensor c: cell state vector
        :param tensor w_x: input to hidden weights
        :param tensor w_h: hidden to hidden weights
        :param tensor b_x: input to hidden bias
        :param tensor b_h: hidden to hidden bias
        :returns: next hidden, cell states
        """
        gx = torch.addmm(b_x, x, w_x.t())
        gh = torch.addmm(b_h, h, w_h.t())
        xi, xf, xo, xn = gx.chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)
        inputgate = torch.sigmoid(xi + hi)
        forgetgate = torch.sigmoid(xf + hf)
        outputgate = torch.sigmoid(xo + ho)
        newgate = torch.tanh(xn + hn)
        c = forgetgate * c + inputgate * newgate
        h = outputgate * torch.tanh(c)
        return h, c

    #Takes input tensor x with dimensions: [T, B, X]
    def forward(self, x, states):
        h, c = states
        outputs = []
        inputs = x.unbind(0)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.w_x, self.w_h, self.b_x, self.b_h)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

class Linear(nn.Module):
    """Linear layer for language model

    :param int input_size: size of input vector
    :param int hidden_size: size of hidden state vector
    """
    def __init__(self, input_size, hidden_size):
        """Initialize Linear layer"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x):
        z = torch.addmm(self.b, x.view(-1, x.size(2)), self.W.t())
        return z

    def __repr__(self):
        return f"FC(input: {self.input_size}, output: {self.hidden_size})"

class Model(nn.Module):
    """Network class for language model

    :param int vocab_size: size of vocabulary
    :param int hidden_size: size of hidden layer
    :param int layer_num: number of hidden layers
    :param float dropout: drop out rate
    :param float winit: the bound of the uniform distribution
    :param int w_rank: rank of all input to hidden matrices
    :param int u_ranks: rank of all hidden to hidden matrices
    :param string lstm_type: type of lstm
    """

    def __init__(self, vocab_size, hidden_size, layer_num, dropout, winit, \
        w_rank=None,u_ranks=None,lstm_type = "pytorch",device=None):
        """Initialize Network"""
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.winit = winit
        self.lstm_type = lstm_type
        self.embed = Embed(vocab_size, hidden_size)

        if u_ranks is not None and lstm_type !="vm_group":
            u_ranks=u_ranks[-1]

        if lstm_type == "vmgroup":
            self.rnns = [myVMLSTM_Group(hidden_size, hidden_size,\
                w_rank=w_rank,u_ranks=u_ranks) for i in range(layer_num)]
            self.rnns = nn.ModuleList(self.rnns)
        elif lstm_type!="vmlmf":
            self.rnns = [LSTM(hidden_size, hidden_size) if lstm_type == "custom"\
                 else nn.LSTM(hidden_size, hidden_size) for i in range(layer_num)]
            self.rnns = nn.ModuleList(self.rnns)
        else:
            self.rnns = [myVMLSTM(hidden_size, hidden_size,\
                w_rank=w_rank,u_ranks=u_ranks,device=device) for i in range(layer_num)]
            self.rnns = nn.ModuleList(self.rnns)

        self.fc = Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters of network"""
        for param in self.parameters():
            nn.init.uniform_(param, -self.winit, self.winit)

    def state_init(self, batch_size):
        """Initialize state vectors
        
        :param int batch_size: size of batch
        """
        dev = next(self.parameters()).device
        states = [(torch.zeros(batch_size, layer.hidden_size, device = dev), \
            torch.zeros(batch_size, layer.hidden_size, device = dev)) \
                if self.lstm_type in ["custom", "vmlmf", "vmgroup", "hmd"]
                  else (torch.zeros(1, batch_size, layer.hidden_size, device = dev), \
                      torch.zeros(1, batch_size, layer.hidden_size, device = dev)) \
                          for layer in self.rnns]
        return states

    def detach(self, states):
        """Returns a new Tensor, detached from the current graph.

        :returns: hidden, cell states tensor
        """
        return [(h.detach(), c.detach()) for (h,c) in states]

    def forward(self, x, states):
        x = self.embed(x)
        x = self.dropout(x)
        for i, rnn in enumerate(self.rnns):
            x, states[i] = rnn(x, states[i])
            x = self.dropout(x)
        scores = self.fc(x)
        return scores, states
