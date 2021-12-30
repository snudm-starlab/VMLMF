################################################################################
# [VMLMF] Low_rank Matrix Factorization with Vector-Multiplication
# Project: Starlab
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: vmlmf_group.py
# - Cell and network class for VMLMF + Group rank version - general task
#
# Version : 1.0
# Date : Oct 14, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
# pylint: disable=C0103, E1101, C0114, R0902,C0116, R0914, R0913, C0123, W0613, W0102,C0321
"""
====================================
 :mod:`vmlmf_group`
====================================
.. moduleauthor:: Hyojin Jeon  <tarahjjeon@snu.ac.kr>
설명
=====
VMLMF Group 구조의 RNN 셀, 네트워크 구조 관련 모듈입니다.

"""
import torch
from torch import nn

TIME_STEPS = 128
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
RECURRENT_MIN = pow(1 / 2, 1 / TIME_STEPS)

class MyVMLMFCellg2(nn.Module):
    """LSTM Cell of VMLMF_group

    :param input_size: size of input vector
    :param int hidden_size: size of hidden state vector
    :param int w_rank: rank of all input to hidden matrices
    :param int u_ranks: rank of all hidden to hidden matrices
    :param int g: number of groups
    :param list recurrent_init: list for initialize recurrent layer
    :param list hidden_init: list for initialize hidden state
    """
    def __init__(self, input_size, hidden_size, w_rank=None, u_ranks=None, g=2,
     recurrent_init=None,hidden_init=None):
        """Initialize VMLMFCellg2"""
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.w_rank = w_rank
        self.u_ranks = u_ranks
        self.g = g

        self.layers=nn.ParameterDict()

        # vm_vector
        self.layers['dia_x']=nn.Parameter(0.1 * torch.randn([1,input_size]))
        self.layers['dia_h']=nn.Parameter(0.1 * torch.randn([1,hidden_size]))

        # UV of input to hid matrix in LMF
        self.layers['u_x']=nn.Parameter(0.1 * torch.randn([input_size,w_rank]))
        self.layers['v_x'] = nn.Parameter(0.1 * torch.randn([4 * hidden_size, w_rank]))

        # UV of hid to hid matrix in LMF
        for g_idx in range(self.g):
            self.layers[f'u_h_{g_idx}']=nn.Parameter( # weight sharing across 4 gates
                0.1*torch.randn([g,int(hidden_size/g),u_ranks[g_idx]]))
            self.layers[f'v_h_{g_idx}']=nn.Parameter(  # V matrix of each gate
                0.1*torch.randn([g,u_ranks[g_idx],4*int(hidden_size/g)]))

        for vec in ['x','h']:
            self.layers[f'bias_{vec}']= nn.Parameter(torch.ones([1, 4*hidden_size]))

    def __repr__(self):
        return f"LSTM VM Group (input:{self.input_size}, hidden:{self.hidden_size},\
             w_rank:{self.w_rank}, u_ranks:{self.u_ranks}"

    def forward(self, x, hiddenStates):
        dev=next(self.parameters()).device
        (h, c) = hiddenStates
        batch_size=h.shape[0]

        #VM operation
        vm_x=torch.cat([self.layers['dia_x']*x.squeeze(),torch.zeros([h.shape[0],
        self.hidden_size-self.input_size],device=dev)],dim=1) \
            if self.hidden_size>=self.input_size else None
        vm_h=self.layers['dia_h']*h.squeeze()

        #LMF_x operation
        lowered_x=torch.matmul(torch.matmul(x,self.layers['u_x']),self.layers['v_x'].t())
        vm_refined_x=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)
        vm_refined_h=torch.zeros(h.shape[0],4*self.hidden_size,device=dev)
        vm_refined_u_h=self.layers['u_h_0'].view(self.hidden_size,self.u_ranks[0])
        vm_refined_Vh=torch.transpose(self.layers['v_h_0'],1,2).contiguous()

        for gate_idx in range(0,4*self.hidden_size,self.hidden_size):
            vm_refined_x[:,gate_idx:gate_idx+self.input_size]=x*torch.sum(
                (self.layers['u_x']*self.layers['v_x'][gate_idx:gate_idx+self.input_size,:]),dim=1)
            gate_g_idx,gate_g_size=int(gate_idx/self.g),int(self.hidden_size/self.g)
            gate_Vh=vm_refined_Vh[:,gate_g_idx:gate_g_idx+gate_g_size,:].reshape(-1,self.u_ranks[0])
            vm_refined_h[:,gate_idx:gate_idx+self.hidden_size]=\
                h*torch.sum((vm_refined_u_h*gate_Vh),dim=1)

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
            h_op=torch.transpose(h_op,0,1)      #[g,batch_size,h/g]

            u_h=self.layers[f'u_h_{i}']   #[g,h/g,u_ranks]
            h_op=torch.bmm(h_op,u_h)             #[g,batch_size,u_ranks]
            Vh=self.layers[f'v_h_{i}']   #[g,u_ranks,h/g*4]
            h_op=torch.bmm(h_op,Vh)             #[g,batch_size,h/g*4]
            h_op=torch.transpose(h_op,0,1)      #[batch_size,g,h/g*4]
            h_op_chunked=h_op if i==0 else h_op_chunked+h_op

        f_h,i_h,n_h,o_h=h_op_chunked.chunk(4,dim=2)

        f_h=f_h.contiguous().view(batch_size,self.hidden_size)
        i_h=i_h.contiguous().view(batch_size,self.hidden_size)
        n_h=n_h.contiguous().view(batch_size,self.hidden_size)
        o_h=o_h.contiguous().view(batch_size,self.hidden_size)

        gh=self.layers['bias_h']-vm_refined_h
        hf,hi,hn,ho=gh.chunk(4,1)

        hf=hf+f_h; hi=hi+i_h; hn=hn+n_h; ho=ho+o_h

        inputgate = torch.sigmoid(xi + hi+vm_x+vm_h)
        forgetgate = torch.sigmoid(xf + hf+vm_x+vm_h)
        outputgate = torch.sigmoid(xo + ho+vm_x+vm_h)
        newgate = torch.tanh(xn +hn+ vm_x + vm_h)
        c_next = forgetgate * c + inputgate * newgate
        h_next = outputgate * torch.tanh(c_next)
        return h_next, c_next


class MyVMLMFgCellg2(nn.Module):
    """LSTM Cell of VMLMF_group without vm

        - vmlmf group cell without vm module for ablation study

        :param int input_size: size of input vector
        :param int hidden_size: size of hidden state vector
        :param int w_rank: rank of all input to hidden matrices
        :param int u_ranks: rank of all hidden to hidden matrices
        :param int g: number of groups
        :param list recurrent_init: list for initialize recurrent layer
        :param list hidden_init: list for initialize hidden state
    """
    def __init__(self, input_size, hidden_size, w_rank=None, u_ranks=None,\
        g=2, recurrent_init=None,hidden_init=None):
        """Initialize VMLMFgCell"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.w_rank = w_rank
        self.u_ranks = u_ranks
        self.g = g

        self.layers=nn.ParameterDict()

        # UV of input to hid matrix in LMF
        self.layers['u_x']=nn.Parameter(0.1 * torch.randn([input_size,w_rank]))
        self.layers['v_x'] = nn.Parameter(0.1 * torch.randn([4 * hidden_size, w_rank]))

        # UV of hid to hid matrix in LMF
        for g_idx in range(self.g):
            self.layers[f'u_h_{g_idx}']=nn.Parameter(0.1*torch.randn(\
                [g,int(hidden_size/g),u_ranks[g_idx]])) #weight sharing across 4 gates
            self.layers[f'v_h_{g_idx}']=nn.Parameter(0.1*torch.randn(\
                [g,u_ranks[g_idx],4*int(hidden_size/g)])) #V matrix of each gate

        for vec in ['x','h']:
            self.layers[f'bias_{vec}']= nn.Parameter(torch.ones([1, 4*hidden_size]))

    def __repr__(self):
        return f"LSTM VM Group (input:{self.input_size}, hidden:{self.hidden_size},\
            w_rank:{self.w_rank}, u_ranks:{self.u_ranks})"

    def forward(self, x, hiddenStates):
        (h, c) = hiddenStates
        batch_size=h.shape[0]

        #LMF_x operation
        lowered_x=torch.matmul(torch.matmul(x,self.layers['u_x']),self.layers['v_x'].t())
        gx=lowered_x+self.layers['bias_x']
        xf, xi, xn, xo = gx.chunk(4, 1)

        #LMF_h operation
        ## 1. LMF for each group
        ## 2. compute redundant values from g0

        index = list(range(self.g))
        # h for each group operation
        for i in range(self.g):
            h_op = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index=index[1:]+index[0:1] if i>0 else index
            h_op=h_op[:,index,:] if i > 0 else h_op
            h_op=torch.transpose(h_op,0,1)    #[g,batch_size,h/g]

            u_h=self.layers[f'u_h_{i}'] #[g,h/g,u_ranks]
            h_op=torch.bmm(h_op,u_h)           #[g,batch_size,u_ranks]
            Vh=self.layers[f'v_h_{i}'] #[g,u_ranks,h/g*4]
            h_op=torch.bmm(h_op,Vh)           #[g,batch_size,h/g*4]
            h_op=torch.transpose(h_op,0,1)    #[batch_size,g,h/g*4]
            h_op_chunked=h_op if i==0 else h_op_chunked+h_op

        f_h,i_h,n_h,o_h=h_op_chunked.chunk(4,dim=2)
        f_h=f_h.contiguous().view(batch_size,self.hidden_size)
        i_h=i_h.contiguous().view(batch_size,self.hidden_size)
        n_h=n_h.contiguous().view(batch_size,self.hidden_size)
        o_h=o_h.contiguous().view(batch_size,self.hidden_size)

        gh=self.layers['bias_h']
        hf,hi,hn,ho=gh.chunk(4,1)
        hf=hf+f_h; hi=hi+i_h; hn=hn+n_h; ho=ho+o_h

        inputgate = torch.sigmoid(xi + hi)
        forgetgate = torch.sigmoid(xf + hf)
        outputgate = torch.sigmoid(xo + ho)
        newgate = torch.tanh(xn +hn)
        c_next = forgetgate * c + inputgate * newgate
        h_next = outputgate * torch.tanh(c_next)
        return h_next, c_next
