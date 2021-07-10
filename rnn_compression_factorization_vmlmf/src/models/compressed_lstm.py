################################################################################
# Starlab RNN-compression with factorization method : Lowrank Factorization with vector-multiplication
#
# Author: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# Version : 1.0
# Date : Jul 08, 2021
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
#import seaborn as sns
import matplotlib.pyplot as plt
import ctypes
import time

torch.autograd.set_detect_anomaly(True)

class myVMLSTMCell_NEO_faster(nn.Module): 
    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, recurrent_init=None,
                 hidden_init=None): 
        super(myVMLSTMCell_NEO_faster, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
    
        
        if self.wRank is not None:
            self.Wu= nn.Parameter(0.1 * torch.randn([input_size, wRank]))
        if self.uRanks is not None:
            self.Uu= nn.Parameter(0.1 * torch.randn([hidden_size,uRanks]))

        wrow = self.input_size if self.wRank is None else self.wRank
        urow = self.hidden_size if self.uRanks is None else self.uRanks

        self.W_dia_vec = nn.Parameter(0.1*torch.randn([1,input_size]))
        self.U_dia_vec = nn.Parameter(0.1*torch.randn([1,hidden_size]))

        # forget gate
        self.Wf = nn.Parameter(0.1 * torch.randn([wrow, hidden_size]))
        self.Uf = nn.Parameter(0.1 * torch.randn([urow, hidden_size]))
        self.bias_f = nn.Parameter(torch.ones([1, hidden_size]))
        # input gate
        self.Wi = nn.Parameter(0.1 * torch.randn([wrow, hidden_size]))
        self.Ui = nn.Parameter(0.1 * torch.randn([urow, hidden_size]))
        self.bias_i = nn.Parameter(torch.ones([1, hidden_size]))
        # cell gate
        self.Wc = nn.Parameter(0.1 * torch.randn([wrow, hidden_size]))
        self.Uc = nn.Parameter(0.1 * torch.randn([urow, hidden_size]))
        self.bias_c = nn.Parameter(torch.ones([1, hidden_size]))
        # output gate
        self.Wo = nn.Parameter(0.1 * torch.randn([wrow, hidden_size]))
        self.Uo = nn.Parameter(0.1 * torch.randn([urow, hidden_size]))
        self.bias_o = nn.Parameter(torch.ones([1, hidden_size]))
        #diagonal_vector
        self.dia_w=0;self.dia_u=0
        #LMF for right
        self.ux=0
        self.uh=0
        #navigator
        self.cnt=0      

    def gate_operation(self, x, h,wv,uv, b,padding=None):
        
        ### vector elementwise-multiplication   
        #x_state_dia = torch.cat([w_dia * x.squeeze(), padding],dim=1) if padding is not None else w_dia * x.squeeze()  
        #h_state_dia = u_dia * h.squeeze()
        x_state_dia=self.dia_w
        h_state_dia=self.dia_u

        ###LMF
        #x_state=torch.matmul(torch.matmul(x,wu),wv)
        #h_state=torch.matmul(torch.matmul(h,uu),uv)
        x_state=torch.matmul(self.ux,wv)
        h_state=torch.matmul(self.uh,uv)

        #avoid diagonal_over-calculated 
        #print("uv shape",uv.shape,wv.shape)
        # print(x.shape, h.shape)
        # print(torch.sum((wu*wv[:,self.input_size].T),dim=1).shape)
        result_w=x*(torch.sum((self.Wu*wv[:,:self.input_size].T),dim=1))
        result_u=h*(torch.sum((self.Uu*uv.T),dim=1))
        # print("0--->", result_w.shape)
        result_w=torch.cat([result_w,padding],dim=1) if padding is not None else result_w
        if self.cnt==0:
            print("x_state_dia {}+ x_state {}+ h_state_dia {}+ h_state {}+ b{} -results_u {}-results_w{}".format(x_state_dia.shape, x_state.shape, h_state_dia.shape,h_state.shape, b.shape,result_u.shape,result_w.shape))
            self.cnt+=1
        return x_state_dia+ x_state + h_state_dia + h_state + b - result_w - result_u

    def forward(self, x, hidden_states, device):
        # step 01. diagonal elements vector * x vector element-wise multiplication
        # step 02. off diagonal elements low rank approximation * x vector multiplication
        # step 03. add 2 vectors from previous process
        (h, c) = hidden_states
        batch_size = h.shape[0]
        paddingTensor = torch.zeros([batch_size, self.hidden_size - self.input_size]).to(
        device) if self.hidden_size - self.input_size >0 else None
        
        self.dia_w=torch.cat([self.W_dia_vec * x.squeeze(), paddingTensor],dim=1) if paddingTensor is not None else self.W_dia_vec * x.squeeze()
        self.dia_u=self.U_dia_vec * h.squeeze()
        
        self.ux=torch.matmul(x,self.Wu)
        self.uh=torch.matmul(h,self.Uu)

        c_next = torch.sigmoid(self.gate_operation(x, h,self.Wf, self.Uf,self.bias_f,paddingTensor)) * c + torch.sigmoid(
            self.gate_operation(x, h, self.Wi, self.Ui, self.bias_i, paddingTensor)) * torch.tanh(
            self.gate_operation(x, h, self.Wc, self.Uc, self.bias_c,paddingTensor))
        h_next = torch.sigmoid(self.gate_operation(x, h, self.Wo, self.Uo, self.bias_o, paddingTensor)) * torch.tanh(c_next)

        
        return h_next, c_next
### 1. VM LMF LSTM Cell 
class myVMLSTMCell_NEO3(nn.Module): 
    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, recurrent_init=None,
                 hidden_init=None): 
        super(myVMLSTMCell_NEO3, self).__init__()
       
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
    
        
        if self.wRank is not None:
            self.Wu_f= nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.Wu_i= nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.Wu_c= nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.Wu_o= nn.Parameter(0.1 * torch.randn([input_size, wRank]))
        if self.uRanks is not None:
            self.Uu_f = nn.Parameter(0.1 * torch.randn([hidden_size, uRanks]))
            self.Uu_i = nn.Parameter(0.1 * torch.randn([hidden_size, uRanks]))
            self.Uu_c = nn.Parameter(0.1 * torch.randn([hidden_size, uRanks]))
            self.Uu_o = nn.Parameter(0.1 * torch.randn([hidden_size, uRanks]))

        wrow = self.input_size if self.wRank is None else self.wRank
        urow = self.hidden_size if self.uRanks is None else self.uRanks

        # forget gate
        self.W_dia_vec = nn.Parameter(0.1*torch.randn([1,input_size]))
        self.U_dia_vec = nn.Parameter(0.1*torch.randn([1,hidden_size]))

        self.Wf = nn.Parameter(0.1 * torch.randn([wrow, hidden_size]))
        self.Uf = nn.Parameter(0.1 * torch.randn([urow, hidden_size]))
        self.bias_f = nn.Parameter(torch.ones([1, hidden_size]))
        # input gate
        self.Wi = nn.Parameter(0.1 * torch.randn([wrow, hidden_size]))
        self.Ui = nn.Parameter(0.1 * torch.randn([urow, hidden_size]))
        self.bias_i = nn.Parameter(torch.ones([1, hidden_size]))
        # cell gate
        self.Wc = nn.Parameter(0.1 * torch.randn([wrow, hidden_size]))
        self.Uc = nn.Parameter(0.1 * torch.randn([urow, hidden_size]))
        self.bias_c = nn.Parameter(torch.ones([1, hidden_size]))
        # output gate
        self.Wo = nn.Parameter(0.1 * torch.randn([wrow, hidden_size]))
        self.Uo = nn.Parameter(0.1 * torch.randn([urow, hidden_size]))
        self.bias_o = nn.Parameter(torch.ones([1, hidden_size]))
        self.cnt=0
        self.cntop=0
        

    def gate_operation(self, x, h,wu, wv,uu, uv, b,w_dia,u_dia,device,padding=None):
        
        self.cntop+=1
        #vector elementwise-multiplication   
        x_state_dia = torch.cat([w_dia * x.squeeze(), padding],dim=1) if padding is not None else w_dia * x.squeeze()  
        h_state_dia = u_dia * h.squeeze()
        
        #LMF
        x_state=torch.matmul(torch.matmul(x,wu),wv)
        h_state=torch.matmul(torch.matmul(h,uu),uv)
        
        #avoid diagonal_over-calculated 
        start_loop=time.time()
        #print("uv shape",uv.shape,wv.shape)
        # print(x.shape, h.shape)
        # print(torch.sum((wu*wv[:,self.input_size].T),dim=1).shape)
        result_w=x*(torch.sum((wu*wv[:,:self.input_size].T),dim=1))
        result_u=h*(torch.sum((uu*uv.T),dim=1))
        result_w=torch.cat([result_w,padding],dim=1) if padding is not None else result_w
    
        return x_state_dia+ x_state + h_state_dia + h_state + b - result_w - result_u



    def forward(self, x, hidden_states, device):
        # step 01. diagonal elements vector * x vector element-wise multiplication
        # step 02. off diagonal elements low rank approximation * x vector multiplication
        # step 03. add 2 vectors from previous process
        (h, c) = hidden_states
        batch_size = h.shape[0]
        paddingTensor = torch.zeros([batch_size, self.hidden_size - self.input_size]).to(
        device) if self.hidden_size - self.input_size >0 else None

        c_next = torch.sigmoid(self.gate_operation(x, h,self.Wu_f, self.Wf,self.Uu_f, self.Uf,self.bias_f,self.W_dia_vec,self.U_dia_vec,device, paddingTensor)) * c + torch.sigmoid(
            self.gate_operation(x, h,self.Wu_i, self.Wi,self.Uu_i, self.Ui, self.bias_i,self.W_dia_vec,self.U_dia_vec,device, paddingTensor)) * torch.tanh(
            self.gate_operation(x, h, self.Wu_c,self.Wc,self.Uu_c, self.Uc, self.bias_c,self.W_dia_vec,self.U_dia_vec,device, paddingTensor))
        h_next = torch.sigmoid(self.gate_operation(x, h, self.Wu_o,self.Wo,self.Uu_o, self.Uo, self.bias_o,self.W_dia_vec,self.U_dia_vec,device, paddingTensor)) * torch.tanh(c_next)

        
        return h_next, c_next

### 2. Vanilla LSTM Cell 
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

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, recurrent_init=None,isMask=False,isBand=True,
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
        

   
        
        return h_next, c_next


class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRanks=None, recurrent=False,cell=myLSTMCell, **kwargs):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRanks = uRanks
        self.drop=nn.Dropout(p=0.5)
        self.cell=cell
        
        self.uRanks=uRanks[0] if type(uRanks) is list else uRanks
        #print()
        #print("# 1. cell:{}".format(self.cell))
        #print("# 2. wRank:{} / uRank:{}".format(wRank,uRanks))

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
            rnn_cells.append(self.cell(in_size, hidden_size, wRank=self.wRank, uRanks=self.uRanks, **kwargs))
            in_size = hidden_size

        self.rnncells = nn.ModuleList(rnn_cells)

       
        
    def forward(self, x, hidden=None):
        time_index = self.time_index
        batch_index = self.batch_index
        hiddens = []

        i = 0
        for cell in self.rnncells:
           
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
    
class Net(nn.Module):
    def __init__(self, input_size, layer_sizes=[32, 32], wRank=None, uRanks=None, model=myLSTM,cell=myLSTMCell,g=None):
        super(Net, self).__init__()
        recurrent_inits = []
        self.cell=cell
        n_layer = len(layer_sizes) + 1
        for _ in range(n_layer - 1):
            recurrent_inits.append(
                lambda w: nn.init.uniform_(w, 0, RECURRENT_MAX)
            )
        recurrent_inits.append(lambda w: nn.init.uniform_(
            w, RECURRENT_MIN, RECURRENT_MAX))
        self.rnn = model(
            input_size, hidden_layer_sizes=layer_sizes,
            batch_first=True, recurrent_inits=recurrent_inits,
            wRank=wRank, uRanks=uRanks,cell=self.cell
        )
        self.lin = nn.Linear(layer_sizes[-1], 18)

        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

    def forward(self, x, hidden=None):
        y, _ = self.rnn(x, hidden)
        return self.lin(y[:, -1]).squeeze(1)
