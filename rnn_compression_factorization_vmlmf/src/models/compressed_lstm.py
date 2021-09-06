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

torch.autograd.set_detect_anomaly(False)

class myVMLSTMCell_NEO_final_group(nn.Module):
    

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, g=2, recurrent_init=None,hidden_init=None):
        super(myVMLSTMCell_NEO_final_group, self).__init__()
        print("last updated 0825 21:00"        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
        self.g = g

        self.layers=nn.ParameterDict()
        
        # vm_vector
        self.layers['dia_x']=nn.Parameter(0.1 * torch.randn([1,input_size]))
        self.layers['dia_h']=nn.Parameter(0.1 * torch.randn([1,hidden_size]))
        
        # UV of input to hid matrix in LMF
        self.layers['Ux']=nn.Parameter(0.1 * torch.randn([input_size,wRank]))
        self.layers['Vx'] = nn.Parameter(0.1 * torch.randn([4 * hidden_size, wRank]))

        # UV of hid to hid matrix in LMF
        for g_idx in range(self.g):
            self.layers['Uh_{}'.format(g_idx)]=nn.Parameter(0.1*torch.randn([g,int(hidden_size/g),uRanks[g_idx]])) # weight sharing across 4 gates
            self.layers['Vh_{}'.format(g_idx)]=nn.Parameter(0.1*torch.randn([g,uRanks[g_idx],4*int(hidden_size/g)])) # V matrix of each gate  

        for vec in ['x','h']:
            self.layers['bias_{}'.format(vec)]= nn.Parameter(torch.ones([1, 4*hidden_size]))
    
    def __repr__(self):
        return "LSTM VM Group (input:{}, hidden:{}, wRank:{}, uRanks:{})".format(self.input_size,self.hidden_size,self.wRank,self.uRanks)

    def forward(self, x, hiddenStates,device):

        dev=next(self.parameters()).device
        (h, c) = hiddenStates
        batch_size=h.shape[0]

        #VM operation
        vm_x=torch.cat([self.layers['dia_x']*x.squeeze(),torch.zeros([h.shape[0],self.hidden_size-self.input_size],device=dev)],dim=1) if self.hidden_size>=self.input_size else None
        vm_h=self.layers['dia_h']*h.squeeze()

        #LMF_x operation 
        lowered_x=torch.matmul(torch.matmul(x,self.layers['Ux']),self.layers['Vx'].t())
        ## x and h.shape[0] == batch_size 81 
        vm_refined_x=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)
        vm_refined_h=torch.zeros(h.shape[0],4*self.hidden_size,device=dev)
        #print(f"Uh shape:{self.layers['Uh_0'].shape}")
        vm_refined_Uh=self.layers['Uh_0'].view(self.hidden_size,self.uRanks[0])
        #print(f">>> Vh_0 shape[g,urank,4*h/g] <-> {self.layers['Vh_0'].shape}")
        vm_refined_Vh=torch.transpose(self.layers['Vh_0'],1,2).contiguous()
        #print(f">>> Vh_0 shape[g,urank,4*h/g] not influenced<-> {self.layers['Vh_0'].shape}")
        #print(f">>> vm_refined_Vh shape[g,4*h/g,urank] <-> {vm_refined_Vh.shape}")

        vm_refined_Vh=vm_refined_Vh.view(-1,self.uRanks[0])
        #print(f">>> vm_refined_Vh shape[g,4*h/g,urank] <-> {vm_refined_Vh.shape}")
        for gate_idx in range(0,4*self.hidden_size,self.hidden_size): 
            vm_refined_x[:,gate_idx:gate_idx+self.input_size]=x*torch.sum((self.layers['Ux']*self.layers['Vx'][gate_idx:gate_idx+self.input_size,:]),dim=1)
            vm_refined_h[:,gate_idx:gate_idx+self.hidden_size]=h*torch.sum((vm_refined_Uh*vm_refined_Vh[gate_idx:gate_idx+self.hidden_size,:]),dim=1)

        gx=lowered_x-vm_refined_x+self.layers['bias_x']
        xi, xf, xo, xn = gx.chunk(4, 1)
        

        #LMF_h operation
        ## 1. LMF for each group
        ## 2. compute redundant values from g0
        
        index = list(range(self.g))
        # h for each group operation
        
        for i in range(self.g):
            h_op = h.view(batch_size, self.g, int(self.hidden_size / self.g))  
            index=index[1:]+index[0:1] if i>0 else index
            h_op=h_op[:,index,:] if i > 0 else h_op
            h_op=torch.transpose(h_op,0,1) #[g,batch_size,h/g]

            Uh=self.layers['Uh_{}'.format(i)] #[g,h/g,uRanks]
            h_op=torch.bmm(h_op,Uh) #[g,batch_size,uRanks]
            Vh=self.layers['Vh_{}'.format(i)] #[g,uRanks,h/g*4]
            h_op=torch.bmm(h_op,Vh) #[g,batch_size,h/g*4]
            h_op=torch.transpose(h_op,0,1) #[batch_size,g,h/g*4]
            g_h=h_op.contiguous().view(batch_size,self.hidden_size*4) if i ==0 else g_h+h_op.contiguous().view(batch_size,self.hidden_size*4)

        gh=g_h-vm_refined_h+self.layers['bias_h'] 
        hf,hi,hn,ho=gh.chunk(4,1)

        inputgate = torch.sigmoid(xi + hi+vm_x+vm_h)
        forgetgate = torch.sigmoid(xf + hf+vm_x+vm_h)
        outputgate = torch.sigmoid(xo + ho+vm_x+vm_h)
        newgate = torch.tanh(xn +hn+ vm_x + vm_h)
        c_next = forgetgate * c + inputgate * newgate
        h_next = outputgate * torch.tanh(c_next)
        return h_next, c_next

class myVMLSTMCELL_NEO_final(nn.Module): 
    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, recurrent_init=None,
                 hidden_init=None): 
        super(myVMLSTMCELL_NEO_final, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.dropout = dropout

        self.wRank=wRank
        self.uRanks=uRanks

        #U in LMF
        self.U_x=nn.Parameter(0.1 * torch.randn([input_size,wRank]))
        self.U_h=nn.Parameter(0.1 * torch.randn([hidden_size,uRanks]))

        #V in LMF
        self.V_x = nn.Parameter(0.1 * torch.randn([4 * hidden_size, wRank]))
        self.V_h = nn.Parameter(0.1 * torch.randn([4 * hidden_size, uRanks]))

        #bias
        self.b_x = nn.Parameter(0.1 * torch.randn([4 * hidden_size]))
        self.b_h = nn.Parameter(0.1 * torch.randn([4 * hidden_size]))

        #diagonal vector
        self.dia_x=nn.Parameter(0.1 * torch.randn([1,input_size]))
        self.dia_h=nn.Parameter(0.1 * torch.randn([1,hidden_size]))

        #save navigator
        self.cnt=0

    def __repr__(self):
        return "LSTM_FINAL(input: {}, hidden: {}, wRank: {}, uRanks: {})".format(self.input_size, self.hidden_size,self.wRank,self.uRanks)
    def forward(self, x, hidden_states,device):
        # step 01. diagonal elements vector * x vector element-wise multiplication
        # step 02. off diagonal elements low rank approximation * x vector multiplication
        # step 03. add 2 vectors from previous process
        dev = next(self.parameters()).device

        (h, c) = hidden_states
        #save vm - redundant values
        vm_refined_x=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)
        vm_refined_h=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)

        #vm (for all 4 gates)
        vm_x=torch.cat([self.dia_x*x.squeeze(),torch.zeros([h.shape[0],self.hidden_size-self.input_size],device=dev)],dim=1) if self.hidden_size>=self.input_size else None
        vm_h=self.dia_h*h.squeeze()
        """vm_x=torch.cat([vm_x for _ in range(4)],dim=1)
        vm_h=torch.cat([vm_h for _ in range(4)],dim=1)"""
       
        #lmf
        lowered_x=torch.matmul(torch.matmul(x,self.U_x),self.V_x.t())
        lowered_h=torch.matmul(torch.matmul(h,self.U_h),self.V_h.t())

        for gate_idx in range(0,4*self.hidden_size,self.hidden_size):
            vm_refined_x[:,gate_idx:gate_idx+self.input_size]=x*torch.sum((self.U_x*self.V_x[gate_idx:gate_idx+self.input_size,:]),dim=1)
            vm_refined_h[:,gate_idx:gate_idx+self.hidden_size]=h*torch.sum((self.U_h*self.V_h[gate_idx:gate_idx+self.hidden_size,:]),dim=1)
        gx=lowered_x-vm_refined_x+self.b_x
        gh=lowered_h-vm_refined_h+self.b_h
        xi, xf, xo, xn = gx.chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)
        
        inputgate = torch.sigmoid(xi + hi+vm_x+vm_h)
        forgetgate = torch.sigmoid(xf + hf+vm_x+vm_h)
        outputgate = torch.sigmoid(xo + ho+vm_x+vm_h)
        newgate = torch.tanh(xn + hn+vm_x+vm_h)
        c_next = forgetgate * c + inputgate * newgate
        h_next = outputgate * torch.tanh(c_next)
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
