################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
# Project: Starlab 
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: vmlmf.py
# - Cell and network class for VMLMF - general task
#
# Version : 1.0
# Date : Oct 14, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

TIME_STEPS = 128
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
RECURRENT_MIN = pow(1 / 2, 1 / TIME_STEPS)

class myVMLMF_CELL(nn.Module): 
    """
    LSTM Cell of VMLMF
        - vmmodule for complement information loss 
        - LMF for reducing model size

    @params wRank
        rank of all W matrices
    @params uRank
        rank of all U matrices
    """
    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, recurrent_init=None,hidden_init=None): 
        super(myVMLMF_CELL, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.wRank=wRank
        self.uRanks=uRanks

        #U in LMF
        self.U_x=nn.Parameter(0.1 * torch.randn([input_size,wRank]))
        self.U_h=nn.Parameter(0.1 * torch.randn([hidden_size,uRanks]))

        #V in LMF
        self.V_x = nn.Parameter(0.1 * torch.randn([4 * hidden_size, wRank]))   #four gates of LSTM
        self.V_h = nn.Parameter(0.1 * torch.randn([4 * hidden_size, uRanks]))  #four gates of LSTM

        #bias
        self.b_x = nn.Parameter(0.1 * torch.randn([4 * hidden_size]))          #four gates of LSTM
        self.b_h = nn.Parameter(0.1 * torch.randn([4 * hidden_size]))          #four gates of LSTM

        #diagonal vector
        self.dia_x=nn.Parameter(0.1 * torch.randn([1,input_size]))
        self.dia_h=nn.Parameter(0.1 * torch.randn([1,hidden_size]))

        #save navigator
        self.cnt=0

    def __repr__(self):
        return "LSTM_FINAL(input: {}, hidden: {}, wRank: {}, uRanks: {})".format(self.input_size, self.hidden_size,self.wRank,self.uRanks)
    
    def forward(self, x, hidden_states):
        # step 01. diagonal elements vector * x vector element-wise multiplication
        # step 02. off diagonal elements low rank approximation * x vector multiplication
        # step 03. add 2 vectors from previous process
        dev = next(self.parameters()).device

        #hidden states from previous time step (h_{t-1}, c_{t-1})
        (h, c) = hidden_states
        #save vm - redundant values
        vm_refined_x=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)
        vm_refined_h=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)

        #vm (for all 4 gates)
        vm_x=torch.cat([self.dia_x*x.squeeze(),torch.zeros([h.shape[0],self.hidden_size-self.input_size],device=dev)],dim=1) if self.hidden_size>=self.input_size else None #to only use diagonal elements.
        vm_h=self.dia_h*h.squeeze()
       
        #lmf
        lowered_x=torch.matmul(torch.matmul(x,self.U_x),self.V_x.t())
        lowered_h=torch.matmul(torch.matmul(h,self.U_h),self.V_h.t())
        
        #compute compressed vm (erase diagonal element*input vector from the results of lmf)
        for gate_idx in range(0,4*self.hidden_size,self.hidden_size):
            vm_refined_x[:,gate_idx:gate_idx+self.input_size]=x*torch.sum((self.U_x*self.V_x[gate_idx:gate_idx+self.input_size,:]),dim=1)
            vm_refined_h[:,gate_idx:gate_idx+self.hidden_size]=h*torch.sum((self.U_h*self.V_h[gate_idx:gate_idx+self.hidden_size,:]),dim=1)
        
        #sum up the values from three operations
        gx=lowered_x-vm_refined_x+self.b_x
        gh=lowered_h-vm_refined_h+self.b_h

        #divide results into 4, for four gates
        xi, xf, xo, xn = gx.chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)
        
        #gate operations
        inputgate = torch.sigmoid(xi + hi+vm_x+vm_h)
        forgetgate = torch.sigmoid(xf + hf+vm_x+vm_h)
        outputgate = torch.sigmoid(xo + ho+vm_x+vm_h)
        newgate = torch.tanh(xn + hn+vm_x+vm_h)

        #hidden and cell states for present time step (h_t, c_t)
        c_next = forgetgate * c + inputgate * newgate
        h_next = outputgate * torch.tanh(c_next)
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

    - four gate in LSTM share W and U
    '''

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, recurrent_init=None,hidden_init=None):
        super(myLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks 
        if type(self.uRanks) is list:
            self.uRanks=uRanks[0]
        
        if wRank is None:   #for four gates in vanilla LSTM
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W4 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:               #for four gates in Low-rank LSTM
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W4 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRanks is None:   #for four gates in vanilla LSTM
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U4 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:                #for four gates in Low-rank LSTM   
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

        #hidden and cell states for previous time step (h_{t-1}, c_{t-1})
        (h, c) = hiddenStates

        if self.wRank is None:      #Vanilla LSTM
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
            wVal4 = torch.matmul(x, self.W4)
        else:                       #Low-rank LSTM
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)
            wVal4 = torch.matmul(
                torch.matmul(x, self.W), self.W4)

        if self.uRanks is None:     #Vanilla LSTM
            uVal1 = torch.matmul(h, self.U1)
            uVal2 = torch.matmul(h, self.U2)
            uVal3 = torch.matmul(h, self.U3)
            uVal4 = torch.matmul(h, self.U4)
        else:                       #Low-rank LSTM
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

        #gate operations
        i = torch.sigmoid(matVal_i + self.bias_i)
        f = torch.sigmoid(matVal_f + self.bias_f)
        o = torch.sigmoid(matVal_o + self.bias_o)
        c_tilda = torch.tanh(matVal_c + self.bias_c)

        c_next = f * c + i * c_tilda
        h_next = o * torch.tanh(c_next)

        #hidden and cell states for present time step (h_t, c_t)
        return h_next, c_next


class myLSTM(nn.Module):
    """
    LSTM layer connecting LSTM Cells
    @param cell 
        type of LSTM cells : vanilla, low-rank, vmlmf
    @params wRank
        rank of all W matrices
    @params uRank
        rank of all U matrices
    """
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRanks=None, recurrent=False,cell=myLSTMCell, **kwargs):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRanks = uRanks[-1] if type(uRanks) == list and len(uRanks)<2 else uRanks
        self.drop=nn.Dropout(p=0.5)
        self.cell=cell
        
        self.uRanks=uRanks[0] if type(uRanks) is list and len(uRanks) < 2 else uRanks
       
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
                h, c = cell(x_time[t], (h, c))
                outputs.append(h)

            x = torch.stack(outputs, time_index)
            hiddens.append(h)
            i = i + 1

        return x, torch.cat(hiddens, -1)
    
class Net(nn.Module):
    """
    Uni-directional LSTM Network 
        LSTM layer + Linear layer
        uniform distribution initialization
    @param cell 
        type of LSTM cells : vanilla, low-rank, vmlmf
    @params wRank
        rank of all W matrices
    @params uRank
        rank of all U matrices
    """
    def __init__(self, input_size, layer_sizes=[32, 32], wRank=None, uRanks=None, model=myLSTM,cell=myLSTMCell,g=None):
        super(Net, self).__init__()
        recurrent_inits = []
        
        n_layer = len(layer_sizes) + 1
        
        for _ in range(n_layer - 1):
            recurrent_inits.append(lambda w: nn.init.uniform_(w, 0, RECURRENT_MAX))
        recurrent_inits.append(lambda w: nn.init.uniform_(w, RECURRENT_MIN, RECURRENT_MAX))
        self.rnn = model(
            input_size, hidden_layer_sizes=layer_sizes,
            batch_first=True, recurrent_inits=recurrent_inits,
            wRank=wRank, uRanks=uRanks,cell=cell
        )
        self.lin = nn.Linear(layer_sizes[-1], 18)
        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

        ## for unit_test
        self.cell=cell(input_size,layer_sizes[-1],wRank=wRank,uRanks=uRanks)
    def forward(self, x, hidden=None):
        y, _ = self.rnn(x, hidden)
        return self.lin(y[:, -1]).squeeze(1)
