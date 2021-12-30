################################################################################
# [VMLMF] Low_rank Matrix Factorization with Vector-Multiplication
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
# pylint: disable=C0103, E1101, C0114, R0902,C0116, R0914, R0913, C0123, W0613, W0102
"""
====================================
 :mod:`vmlmf`
====================================
.. moduleauthor:: Hyojin Jeon  <tarahjjeon@snu.ac.kr>
설명
=====
VMLMF의 RNN 셀, 네트워크 구조 관련 모듈입니다.

"""
from torch import nn
import torch

TIME_STEPS = 128
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
RECURRENT_MIN = pow(1 / 2, 1 / TIME_STEPS)


class MyVMLMFCell(nn.Module):
    """LSTM Cell of VMLMF

    :param int input_size: size of input vector
    :param int hidden_size: size of hidden state vector
    :param int w_rank: rank of all input to hidden matrices
    :param int u_ranks: rank of all hidden to hidden matrices
    """

    def __init__(self, input_size, hidden_size, recurrent_init=None,w_rank=None, u_ranks=None):
        """Initialize VMLMFCell"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_rank = w_rank
        self.u_ranks = u_ranks[-1]

        # U in LMF
        self.u_x = nn.Parameter(0.1 * torch.randn([input_size, w_rank]))
        self.u_h = nn.Parameter(0.1 * torch.randn([hidden_size, u_ranks]))

        # V in LMF
        self.v_x = nn.Parameter(0.1 * torch.randn([4 * hidden_size, w_rank]))  # four gates of LSTM
        self.v_h = nn.Parameter(0.1 * torch.randn([4 * hidden_size, u_ranks]))  # four gates of LSTM

        # bias
        self.b_x = nn.Parameter(0.1 * torch.randn([4 * hidden_size]))  # four gates of LSTM
        self.b_h = nn.Parameter(0.1 * torch.randn([4 * hidden_size]))  # four gates of LSTM

        # diagonal vector
        self.dia_x = nn.Parameter(0.1 * torch.randn([1, input_size]))
        self.dia_h = nn.Parameter(0.1 * torch.randn([1, hidden_size]))

        # save navigator
        self.cnt = 0

    def __repr__(self):
        return f"LSTM_FINAL(input: {self.input_size}, hidden: {self.hidden_size}, " \
               f"w_rank: {self.w_rank}, u_ranks: {self.u_ranks})"

    def forward(self, x, hidden_states):
        # step 01. diagonal elements vector & x vector element-wise multiplication
        # step 02. off diagonal elements low rank approximation * x vector
        # step 03. add 2 vectors from previous process
        dev = next(self.parameters()).device

        # hidden states from previous time step (h_{t-1}, c_{t-1})
        (h, c) = hidden_states
        # save vm - redundant values
        vm_refined_x = torch.zeros(x.shape[0], 4 * self.hidden_size, device=dev)
        vm_refined_h = torch.zeros(x.shape[0], 4 * self.hidden_size, device=dev)

        # vm (for all 4 gates)
        vm_x = torch.cat([self.dia_x * x.squeeze(), torch.zeros(
            [h.shape[0], self.hidden_size - self.input_size], device=dev)],
            dim=1) if self.hidden_size >= self.input_size else None
        vm_h = self.dia_h * h.squeeze()

        # lmf
        lowered_x = torch.matmul(torch.matmul(x, self.u_x), self.v_x.t())
        lowered_h = torch.matmul(torch.matmul(h, self.u_h), self.v_h.t())

        # compute compressed vm (erase diagonal element*input vector from the results of lmf)
        for gate_idx in range(0, 4 * self.hidden_size, self.hidden_size):
            vm_refined_x[:, gate_idx:gate_idx + self.input_size] = x * torch.sum(
                (self.u_x * self.v_x[gate_idx:gate_idx + self.input_size, :]), dim=1)
            vm_refined_h[:, gate_idx:gate_idx + self.hidden_size] = h * torch.sum(
                (self.u_h * self.v_h[gate_idx:gate_idx + self.hidden_size, :]), dim=1)

        # sum up the values from three operations
        gx = lowered_x - vm_refined_x + self.b_x
        gh = lowered_h - vm_refined_h + self.b_h

        # divide results into 4, for four gates
        xi, xf, xo, xn = gx.chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)

        # gate operations
        inputgate = torch.sigmoid(xi + hi + vm_x + vm_h)
        forgetgate = torch.sigmoid(xf + hf + vm_x + vm_h)
        outputgate = torch.sigmoid(xo + ho + vm_x + vm_h)
        newgate = torch.tanh(xn + hn + vm_x + vm_h)

        # hidden and cell states for present time step (h_t, c_t)
        c_next = forgetgate * c + inputgate * newgate
        h_next = outputgate * torch.tanh(c_next)
        return h_next, c_next

class MyLSTMCell(nn.Module):
    """LSTM Cell of Vanilla LSTM and LMF

    :param int input_size: size of input vector
    :param int hidden_size: size of hidden state vector
    :param int w_rank: rank of all W matrices
    :param int u_rank: rank of all U matrices
    :param list recurrent_init: list to initialize recurrent layers
    :param list hidden_init: list to initialize hidden state
    """
    def __init__(self, input_size, hidden_size, w_rank=None, u_ranks=None,
                 recurrent_init=None, hidden_init=None):
        """Initialize LSTMCell"""
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.w_rank = w_rank
        self.u_ranks = u_ranks
        if type(self.u_ranks) is list:
            self.u_ranks = u_ranks[0]

        if w_rank is None:  # for four gates in vanilla LSTM
            self.w1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.w2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.w3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.w4 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:  # for four gates in Low-rank LSTM
            self.w = nn.Parameter(0.1 * torch.randn([input_size, w_rank]))
            self.w1 = nn.Parameter(0.1 * torch.randn([w_rank, hidden_size]))
            self.w2 = nn.Parameter(0.1 * torch.randn([w_rank, hidden_size]))
            self.w3 = nn.Parameter(0.1 * torch.randn([w_rank, hidden_size]))
            self.w4 = nn.Parameter(0.1 * torch.randn([w_rank, hidden_size]))

        if u_ranks is None:  # for four gates in vanilla LSTM
            self.u1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.u2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.u3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.u4 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:  # for four gates in Low-rank LSTM
            self.u = nn.Parameter(0.1 * torch.randn([hidden_size, u_ranks]))
            self.u1 = nn.Parameter(0.1 * torch.randn([u_ranks, hidden_size]))
            self.u2 = nn.Parameter(0.1 * torch.randn([u_ranks, hidden_size]))
            self.u3 = nn.Parameter(0.1 * torch.randn([u_ranks, hidden_size]))
            self.u4 = nn.Parameter(0.1 * torch.randn([u_ranks, hidden_size]))

        self.bias_f = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_i = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_c = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_o = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, hiddenStates):

        # hidden and cell states for previous time step (h_{t-1}, c_{t-1})
        (h, c) = hiddenStates

        if self.w_rank is None:  # Vanilla LSTM
            wVal1 = torch.matmul(x, self.w1)
            wVal2 = torch.matmul(x, self.w2)
            wVal3 = torch.matmul(x, self.w3)
            wVal4 = torch.matmul(x, self.w4)
        else:  # Low-rank LSTM
            wVal1 = torch.matmul(
                torch.matmul(x, self.w), self.w1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.w), self.w2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.w), self.w3)
            wVal4 = torch.matmul(
                torch.matmul(x, self.w), self.w4)

        if self.u_ranks is None:  # Vanilla LSTM
            uVal1 = torch.matmul(h, self.u1)
            uVal2 = torch.matmul(h, self.u2)
            uVal3 = torch.matmul(h, self.u3)
            uVal4 = torch.matmul(h, self.u4)
        else:  # Low-rank LSTM
            uVal1 = torch.matmul(
                torch.matmul(h, self.u), self.u1)
            uVal2 = torch.matmul(
                torch.matmul(h, self.u), self.u2)
            uVal3 = torch.matmul(
                torch.matmul(h, self.u), self.u3)
            uVal4 = torch.matmul(
                torch.matmul(h, self.u), self.u4)

        matVal_i = wVal1 + uVal1
        matVal_f = wVal2 + uVal2
        matVal_o = wVal3 + uVal3
        matVal_c = wVal4 + uVal4

        # gate operations
        i = torch.sigmoid(matVal_i + self.bias_i)
        f = torch.sigmoid(matVal_f + self.bias_f)
        o = torch.sigmoid(matVal_o + self.bias_o)
        c_tilda = torch.tanh(matVal_c + self.bias_c)

        c_next = f * c + i * c_tilda
        h_next = o * torch.tanh(c_next)

        # hidden and cell states for present time step (h_t, c_t)
        return h_next, c_next


class MyLSTM(nn.Module):
    """LSTM layer connecting LSTM Cells

    :param int input_size: size of input vector
    :param int hidden_layer_sizes: size of hidden layers
    :param boolean batch_first:  If True, then the input and output
    tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
    :param int w_rank: rank of all input to hidden matrices
    :param int u_ranks: rank of all hidden to hidden matrices
    :param list recurrent_init: list to initialize recurrent layers
    :param list hidden_init: list to initialize hidden state
    :param Class cell: class of rnn cell implementation
    """

    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True,
                 recurrent_inits=None, hidden_inits=None, w_rank=None, u_ranks=None,
                 cell=MyLSTMCell, **kwargs):
        """Initialize LSTM layer"""
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.w_rank = w_rank
        self.u_ranks = u_ranks[-1] if type(u_ranks) is list and len(u_ranks) < 2 else u_ranks
        self.drop = nn.Dropout(p=0.5)
        self.cell = cell

        self.u_ranks = u_ranks[0] if type(u_ranks) is list and len(u_ranks) < 2 else u_ranks

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
            rnn_cells.append(self.cell(in_size, hidden_size,
                                       w_rank=self.w_rank, u_ranks=self.u_ranks, **kwargs))
            in_size = hidden_size

        self.rnncells = nn.ModuleList(rnn_cells)

    def forward(self, x, hidden=None):
        time_index = self.time_index
        batch_index = self.batch_index
        hiddens = []
        i = 0
        for cell in self.rnncells:
            device = x.device
            h = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(device)
            c = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(device)
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
    """Uni-directional LSTM Network (LSTM layer + Linear layer)

    :param int input_size: size of input vector
    :param int layer_sizes: size of hidden layer
    :param int w_rank: rank of all input to hidden matrices
    :param int u_rank: rank of all hidden to hidden matrices
    :param Class model: class of rnn model implementation
    :param Class cell: class of rnn cell implementation
    """
    def __init__(self, input_size, layer_sizes=[32, 32],
                 w_rank=None, u_rank=None, model=MyLSTM, cell=MyLSTMCell):
        """Initialize Network"""
        super().__init__()
        recurrent_inits = []

        n_layer = len(layer_sizes) + 1
        # self.cell=cell
        for _ in range(n_layer - 1):
            recurrent_inits.append(lambda w: nn.init.uniform_(w, 0, RECURRENT_MAX))
        recurrent_inits.append(lambda w: nn.init.uniform_(w, RECURRENT_MIN, RECURRENT_MAX))
        self.rnn = model(
            input_size, hidden_layer_sizes=layer_sizes,
            batch_first=True, recurrent_inits=recurrent_inits,
            w_rank=w_rank, u_ranks=u_rank, cell=cell  ## self.cell
        )
        self.lin = nn.Linear(layer_sizes[-1], 18)
        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)
        ## for unit_test
        self.cell = cell(input_size, layer_sizes[-1], w_rank=w_rank, u_ranks=u_rank)

    def forward(self, x, hidden=None):
        y, _ = self.rnn(x, hidden)
        return self.lin(y[:, -1]).squeeze(1)
