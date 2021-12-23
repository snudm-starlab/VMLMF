################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vmlmf import myVMLMF_CELL

#Embedding module.
class Embed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, embed_size))

    def forward(self, x):
        return self.W[x]

    def __repr__(self):
        return "Embedding(vocab: {}, embedding: {})".format(self.vocab_size, self.embed_size)

class myVMLSTM_Group(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0, winit = 0.1,wRank=None,uRanks=None,device=None,g=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.g=g

        self.wRank=wRank
        self.uRanks=uRanks

        #In-Hid in LMF
        self.U_x=nn.Parameter(torch.Tensor(input_size,wRank))
        self.W_x = nn.Parameter(torch.Tensor(4 * hidden_size, wRank))
        #Hid-Hid in group_LMF
        self.U_h = nn.ParameterList([nn.Parameter(torch.Tensor(g,int(hidden_size/g),uRanks[g_idx])) for g_idx in range(self.g)])
        self.V_h = nn.ParameterList([ nn.Parameter(torch.Tensor(g,uRanks[g_idx],4 * int(hidden_size/g))) for g_idx in range(self.g)])

        #bias
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))

        #diagonal vector
        self.dia_x=nn.Parameter(torch.Tensor(1,input_size))
        self.dia_h=nn.Parameter(torch.Tensor(1,hidden_size))
        self.cnt=0

    def __repr__(self):
        return "LSTM(input: {}, hidden: {})".format(self.input_size, self.hidden_size)

    def lstm_step(self, x, h, c, W_x, V_h, b_x, b_h):
        dev = next(self.parameters()).device
        
        #save vm - redundant values
        vm_refined_x=torch.zeros(40,4*self.hidden_size,device=dev)
        vm_refined_h=torch.zeros(40,4*self.hidden_size,device=dev)
       
        #vm (for all 4 gates)
        vm_x=self.dia_x*x.squeeze()
        vm_h=self.dia_h*h.squeeze()
        #print(f"h shape:{h.shape}")
        vm_x=torch.cat([vm_x for _ in range(4)],dim=1)
        vm_h=torch.cat([vm_h for _ in range(4)],dim=1)
        
        lowered_x=torch.matmul(torch.matmul(x,self.U_x),self.W_x.t())
        #lowered_h=torch.matmul(torch.matmul(h,self.U_h),self.W_h.t())
        
        batch_size=h.shape[0]
        index=list(range(self.g))
        for partial_h in range(self.g):
            h_top=h.view(-1,self.g,int(self.hidden_size/self.g))
            index=index[1:]+index[0:1] if partial_h>0 else index
            h_top=h_top[:,index,:] if partial_h > 0 else h_top
            h_top=torch.transpose(h_top,0,1) 

            U_dia=self.U_h[partial_h]
            V_dia=self.V_h[partial_h]
            h_top=torch.bmm(h_top,U_dia)

            h_top=torch.bmm(h_top,V_dia)
            h_top=torch.transpose(h_top,0,1)
            h_top=h_top.contiguous().view(-1,self.hidden_size*4)
            lowered_h=h_top if partial_h==0 else h_top + lowered_h

        


        
        #cal redundant values
        hidden_size=self.hidden_size
        input_size=self.input_size
        re_Uh=self.U_h[0].view(self.hidden_size,self.uRanks[0])
        re_Vh=torch.transpose(self.V_h[0],1,2).contiguous().view(4*self.hidden_size,self.uRanks[0])
        
        for gate_idx in range(0,4*hidden_size,hidden_size):
            temp=torch.sum((self.U_x*self.W_x[gate_idx:gate_idx+input_size,:]),dim=1)
            #print(f"vm_refined_x: {vm_refined_x.shape} temp:{temp.shape}/x.shape {x.shape}")
            vm_refined_x[:,gate_idx:gate_idx+input_size]=x*torch.sum((self.U_x*self.W_x[gate_idx:gate_idx+input_size,:]),dim=1)
            vm_refined_h[:,gate_idx:gate_idx+hidden_size]=h*torch.sum((re_Uh*re_Vh[gate_idx:gate_idx+hidden_size,:]),dim=1)
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

    #Takes input tensor x with dimensions: [T, B, X].
    def forward(self, x, states):
        h, c = states
        outputs = []
        inputs = x.unbind(0)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.W_x, self.V_h, self.b_x, self.b_h)
            outputs.append(h)
        return torch.stack(outputs), (h, c)
    
######## <End of custom vmlmf lstm model code > ############

class myVMLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0, winit = 0.1,wRank=None,uRanks=None,device=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.wRank=wRank
        self.uRanks=uRanks

        #U in LMF
        self.U_x=nn.Parameter(torch.Tensor(input_size,wRank))
        self.U_h=nn.Parameter(torch.Tensor(hidden_size,uRanks))

        #V in LMF
        self.W_x = nn.Parameter(torch.Tensor(4 * hidden_size, wRank))
        self.W_h = nn.Parameter(torch.Tensor(4 * hidden_size, uRanks))

        #bias
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))

        #diagonal vector
        self.dia_x=nn.Parameter(torch.Tensor(1,input_size))
        self.dia_h=nn.Parameter(torch.Tensor(1,hidden_size))

        #save navigator
        self.cnt=0


    def __repr__(self):
        return "LSTM(input: {}, hidden: {})".format(self.input_size, self.hidden_size)

    def lstm_step(self, x, h, c, W_x, W_h, b_x, b_h):
        dev = next(self.parameters()).device
        
        #save vm - redundant values
        vm_refined_x=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)
        vm_refined_h=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)
        
        #vm (for all 4 gates)
        vm_x=self.dia_x*x.squeeze()
        vm_h=self.dia_h*h.squeeze()
        vm_x=torch.cat([vm_x for _ in range(4)],dim=1)
        vm_h=torch.cat([vm_h for _ in range(4)],dim=1)
       
        #lmf(x*U -> x'*V) 
        lowered_x=torch.matmul(torch.matmul(x,self.U_x),self.W_x.t())
        lowered_h=torch.matmul(torch.matmul(h,self.U_h),self.W_h.t())
     
        #cal redundant values
        hidden_size=self.hidden_size
        input_size=self.input_size
        for gate_idx in range(0,4*hidden_size,hidden_size):
            vm_refined_x[:,gate_idx:gate_idx+input_size]=x*torch.sum((self.U_x*self.W_x[gate_idx:gate_idx+input_size,:]),dim=1)
            vm_refined_h[:,gate_idx:gate_idx+hidden_size]=h*torch.sum((self.U_h*self.W_h[gate_idx:gate_idx+hidden_size,:]),dim=1)
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

    #Takes input tensor x with dimensions: [T, B, X].
    def forward(self, x, states):
        h, c = states
        outputs = []
        inputs = x.unbind(0)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.W_x, self.W_h, self.b_x, self.b_h)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

#My custom written LSTM module.
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0, winit = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.W_x = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.W_h = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))

    def __repr__(self):
        return "LSTM(input: {}, hidden: {})".format(self.input_size, self.hidden_size)

    def lstm_step(self, x, h, c, W_x, W_h, b_x, b_h):
        gx = torch.addmm(b_x, x, W_x.t())
        gh = torch.addmm(b_h, h, W_h.t())
        xi, xf, xo, xn = gx.chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)
        inputgate = torch.sigmoid(xi + hi)
        forgetgate = torch.sigmoid(xf + hf)
        outputgate = torch.sigmoid(xo + ho)
        newgate = torch.tanh(xn + hn)
        c = forgetgate * c + inputgate * newgate
        h = outputgate * torch.tanh(c)
        return h, c

    #Takes input tensor x with dimensions: [T, B, X].
    def forward(self, x, states):
        h, c = states
        
        outputs = []
        inputs = x.unbind(0)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.W_x, self.W_h, self.b_x, self.b_h)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

class Linear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x):
        #.view() flattens the input which has dimensionality [T,B,X] to dimenstionality [T*B, X].
        z = torch.addmm(self.b, x.view(-1, x.size(2)), self.W.t())
        return z

    def __repr__(self):
        return "FC(input: {}, output: {})".format(self.input_size, self.hidden_size)


class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, layer_num, dropout, winit, wRank=None,uRanks=None,lstm_type = "pytorch",cell=myVMLMF_CELL,device=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.winit = winit
        self.lstm_type = lstm_type
        self.embed = Embed(vocab_size, hidden_size)
        
        if uRanks is not None and lstm_type !="vm_group":
            uRanks=uRanks[-1]
        
        if lstm_type == "vmgroup":
            self.rnns = [myVMLSTM_Group(hidden_size, hidden_size,wRank=wRank,uRanks=uRanks,device=device) for i in range(layer_num)]
            self.rnns = nn.ModuleList(self.rnns)
        elif lstm_type!="vmlmf":
            self.rnns = [LSTM(hidden_size, hidden_size) if lstm_type == "custom" else nn.LSTM(hidden_size, hidden_size) for i in range(layer_num)]
            self.rnns = nn.ModuleList(self.rnns)
        else:
            self.rnns = [myVMLSTM(hidden_size, hidden_size,wRank=wRank,uRanks=uRanks,device=device) for i in range(layer_num)]
            self.rnns = nn.ModuleList(self.rnns)

        self.fc = Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()
        
    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.winit, self.winit)
            
    def state_init(self, batch_size):
        dev = next(self.parameters()).device
        states = [(torch.zeros(batch_size, layer.hidden_size, device = dev), torch.zeros(batch_size, layer.hidden_size, device = dev)) if self.lstm_type == "custom" or self.lstm_type =="vmlmf" or self.lstm_type == "vmgroup" or self.lstm_type == "hmd"
                  else (torch.zeros(1, batch_size, layer.hidden_size, device = dev), torch.zeros(1, batch_size, layer.hidden_size, device = dev)) for layer in self.rnns]
        return states
    
    def detach(self, states):
        return [(h.detach(), c.detach()) for (h,c) in states]
    
    def forward(self, x, states):
        x = self.embed(x)
        x = self.dropout(x)
        for i, rnn in enumerate(self.rnns):
            x, states[i] = rnn(x, states[i])
            x = self.dropout(x)
        scores = self.fc(x)
        return scores, states
