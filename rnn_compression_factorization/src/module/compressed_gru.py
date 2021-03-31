################################################################################
# Starlab RNN-compression with factorization method : Lowrank and group-lowrank rnn
#
# Author: Donghae Jang (jangdonghae@gmail.com), Seoul National University
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


class myGRUCell(nn.Module):
    """
    wRank = rank of W matrix
    (creates 4 matrices if not None else creates 3 matrices)
    uRank = rank of U matrix
    (creates 4 matrices if not None else creates 3 matrices)

    Basic architecture is like:

    r_t = gate_nl(W1x_t + U1h_{t-1} + B_r)
    z_t = gate_nl(W2x_t + U2h_{t-1} + B_g)
    h_t^ = update_nl(W3x_t + r_t*U3(h_{t-1}) + B_h)
    h_t = z_t*h_{t-1} + (1-z_t)*h_t^

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    """

    def __init__(self, input_size, hidden_size, wRank=None, uRank=None, recurrent_init=None,
                 hidden_init=None):
        super(myGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRank = uRank

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))

        self.bias_r = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, h):
        if self.wRank is None:
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
        else:
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)

        if self.uRank is None:
            uVal1 = torch.matmul(h, self.U1)
            uVal2 = torch.matmul(h, self.U2)
        else:
            uVal1 = torch.matmul(
                torch.matmul(h, self.U), self.U1)
            uVal2 = torch.matmul(
                torch.matmul(h, self.U), self.U2)
        matVal_r = wVal1 + uVal1
        matVal_z = wVal2 + uVal2

        r = torch.sigmoid(matVal_r + self.bias_r)
        z = torch.sigmoid(matVal_z + self.bias_gate)

        if self.uRank is None:
            matVal_c = wVal3 + torch.matmul(r * h, self.U3)
        else:
            matVal_c = wVal3 + \
                       torch.matmul(torch.matmul(r * h, self.U), self.U3)

        c_tilda = torch.tanh(matVal_c + self.bias_update)

        h_next = z * h + (1.0 - z) * c_tilda

        return h_next

class myGRUCell_group2(nn.Module):
    """
    wRank = rank of W matrix
    (creates 4 matrices if not None else creates 3 matrices)
    uRank = rank of U matrix
    (creates 4 matrices if not None else creates 3 matrices)

    Basic architecture is like:

    r_t = gate_nl(W1x_t + U1h_{t-1} + B_r)
    z_t = gate_nl(W2x_t + U2h_{t-1} + B_g)
    h_t^ = update_nl(W3x_t + r_t*U3(h_{t-1}) + B_h)
    h_t = z_t*h_{t-1} + (1-z_t)*h_t^

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    """

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, g=2, recurrent_init=None,
                 hidden_init=None):
        super(myGRUCell_group2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
        self.g = g
        #print("wRank is:{}".format(wRank))
        #print("uRank is:{}".format(uRanks))

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRanks is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            # self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            # self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U1_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            # self.U2_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            # self.U3_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            self.U = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[0]]))
            self.U1 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U2 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U3 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))

            self.UU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[1]]))
            self.UU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))

        self.bias_r = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, h):

        index = list(range(self.g))

        #############################
        batch_size = h.shape[0]
        h2 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
        h2 = torch.transpose(h2, 0, 1)
        h2 = torch.bmm(h2, self.U)
        uVal1 = torch.bmm(h2, self.U1)
        uVal2 = torch.bmm(h2, self.U2)
        uVal3 = torch.bmm(h2, self.U3)
        uVal1 = torch.transpose(uVal1, 0, 1)
        uVal2 = torch.transpose(uVal2, 0, 1)
        uVal3 = torch.transpose(uVal3, 0, 1)
        uVal1 = uVal1.contiguous().view(batch_size, self.hidden_size)
        uVal2 = uVal2.contiguous().view(batch_size, self.hidden_size)
        uVal3 = uVal3.contiguous().view(batch_size, self.hidden_size)

        #############################
        h3 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
        index = index[1:] + index[0:1]
        h3 = h3[:, index, :]
        h3 = torch.transpose(h3, 0, 1)
        h3 = torch.bmm(h3, self.UU)
        uuVal1 = torch.bmm(h3, self.UU1)
        uuVal2 = torch.bmm(h3, self.UU2)
        uuVal3 = torch.bmm(h3, self.UU3)
        uuVal1 = torch.transpose(uuVal1, 0, 1)
        uuVal2 = torch.transpose(uuVal2, 0, 1)
        uuVal3 = torch.transpose(uuVal3, 0, 1)
        uuVal1 = uuVal1.contiguous().view(batch_size, self.hidden_size)
        uuVal2 = uuVal2.contiguous().view(batch_size, self.hidden_size)
        uuVal3 = uuVal3.contiguous().view(batch_size, self.hidden_size)

        if self.wRank is None:
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
        else:
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)

        matVal_r = wVal1 + uVal1 + uuVal1
        matVal_z = wVal2 + uVal2 + uuVal2

        r = torch.sigmoid(matVal_r + self.bias_r)
        z = torch.sigmoid(matVal_z + self.bias_gate)

        matVal_c = wVal3 + r * (uVal3 + uuVal3)

        c_tilda = torch.tanh(matVal_c + self.bias_update)

        h_next = z * h + (1.0 - z) * c_tilda

        return h_next


class myGRUCell_group3(nn.Module):
    """
    wRank = rank of W matrix
    (creates 4 matrices if not None else creates 3 matrices)
    uRank = rank of U matrix
    (creates 4 matrices if not None else creates 3 matrices)

    Basic architecture is like:

    r_t = gate_nl(W1x_t + U1h_{t-1} + B_r)
    z_t = gate_nl(W2x_t + U2h_{t-1} + B_g)
    h_t^ = update_nl(W3x_t + r_t*U3(h_{t-1}) + B_h)
    h_t = z_t*h_{t-1} + (1-z_t)*h_t^

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    """

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, g=3, recurrent_init=None,
                 hidden_init=None):
        super(myGRUCell_group3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
        self.g = g
        #print("wRank is:{}".format(wRank))
        #print("uRank is:{}".format(uRanks))

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRanks is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            # self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            # self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U1_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            # self.U2_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            # self.U3_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            self.U = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[0]]))
            self.U1 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U2 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U3 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))

            self.UU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[1]]))
            self.UU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))

            if uRanks[2] > 0:
                self.UUU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[2]]))
                self.UUU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))
                self.UUU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))
                self.UUU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))

        self.bias_r = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, h):

        index = list(range(self.g))

        #############################
        batch_size = h.shape[0]
        h2 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
        h2 = torch.transpose(h2, 0, 1)
        h2 = torch.bmm(h2, self.U)
        uVal1 = torch.bmm(h2, self.U1)
        uVal2 = torch.bmm(h2, self.U2)
        uVal3 = torch.bmm(h2, self.U3)
        uVal1 = torch.transpose(uVal1, 0, 1)
        uVal2 = torch.transpose(uVal2, 0, 1)
        uVal3 = torch.transpose(uVal3, 0, 1)
        uVal1 = uVal1.contiguous().view(batch_size, self.hidden_size)
        uVal2 = uVal2.contiguous().view(batch_size, self.hidden_size)
        uVal3 = uVal3.contiguous().view(batch_size, self.hidden_size)

        #############################
        if self.uRanks[1] > 0:
            h3 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h3 = h3[:, index, :]
            h3 = torch.transpose(h3, 0, 1)
            h3 = torch.bmm(h3, self.UU)
            uuVal1 = torch.bmm(h3, self.UU1)
            uuVal2 = torch.bmm(h3, self.UU2)
            uuVal3 = torch.bmm(h3, self.UU3)
            uuVal1 = torch.transpose(uuVal1, 0, 1)
            uuVal2 = torch.transpose(uuVal2, 0, 1)
            uuVal3 = torch.transpose(uuVal3, 0, 1)
            uuVal1 = uuVal1.contiguous().view(batch_size, self.hidden_size)
            uuVal2 = uuVal2.contiguous().view(batch_size, self.hidden_size)
            uuVal3 = uuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            index = index[1:] + index[0:1]
            uuVal1 = 0
            uuVal2 = 0
            uuVal3 = 0

        #############################
        if self.uRanks[2] > 0:
            h4 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h4 = h4[:, index, :]
            h4 = torch.transpose(h4, 0, 1)
            h4 = torch.bmm(h4, self.UUU)
            uuuVal1 = torch.bmm(h4, self.UUU1)
            uuuVal2 = torch.bmm(h4, self.UUU2)
            uuuVal3 = torch.bmm(h4, self.UUU3)
            uuuVal1 = torch.transpose(uuuVal1, 0, 1)
            uuuVal2 = torch.transpose(uuuVal2, 0, 1)
            uuuVal3 = torch.transpose(uuuVal3, 0, 1)
            uuuVal1 = uuuVal1.contiguous().view(batch_size, self.hidden_size)
            uuuVal2 = uuuVal2.contiguous().view(batch_size, self.hidden_size)
            uuuVal3 = uuuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            uuuVal1 = 0
            uuuVal2 = 0
            uuuVal3 = 0

        if self.wRank is None:
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
        else:
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)

        matVal_r = wVal1 + uVal1 + uuVal1 + uuuVal1
        matVal_z = wVal2 + uVal2 + uuVal2 + uuuVal2

        r = torch.sigmoid(matVal_r + self.bias_r)
        z = torch.sigmoid(matVal_z + self.bias_gate)

        matVal_c = wVal3 + r * (uVal3 + uuVal3 + uuuVal3)

        c_tilda = torch.tanh(matVal_c + self.bias_update)

        h_next = z * h + (1.0 - z) * c_tilda

        return h_next


class myGRUCell_group4(nn.Module):
    """
    wRank = rank of W matrix
    (creates 4 matrices if not None else creates 3 matrices)
    uRank = rank of U matrix
    (creates 4 matrices if not None else creates 3 matrices)

    Basic architecture is like:

    r_t = gate_nl(W1x_t + U1h_{t-1} + B_r)
    z_t = gate_nl(W2x_t + U2h_{t-1} + B_g)
    h_t^ = update_nl(W3x_t + r_t*U3(h_{t-1}) + B_h)
    h_t = z_t*h_{t-1} + (1-z_t)*h_t^

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    """

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, g=4, recurrent_init=None,
                 hidden_init=None):
        super(myGRUCell_group4, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
        self.g = g
        #print("wRank is:{}".format(wRank))
        #print("uRank is:{}".format(uRanks))

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRanks is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            # self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            # self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U1_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            # self.U2_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            # self.U3_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            self.U = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[0]]))
            self.U1 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U2 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U3 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))

            self.UU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[1]]))
            self.UU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))

            if uRanks[2] > 0:
                self.UUU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[2]]))
                self.UUU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))
                self.UUU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))
                self.UUU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))
            if uRanks[3] > 0:
                self.UUUU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[3]]))
                self.UUUU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[3], int(hidden_size / g)]))
                self.UUUU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[3], int(hidden_size / g)]))
                self.UUUU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[3], int(hidden_size / g)]))

        self.bias_r = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, h):

        index = list(range(self.g))

        #############################
        batch_size = h.shape[0]
        h2 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
        h2 = torch.transpose(h2, 0, 1)
        h2 = torch.bmm(h2, self.U)
        uVal1 = torch.bmm(h2, self.U1)
        uVal2 = torch.bmm(h2, self.U2)
        uVal3 = torch.bmm(h2, self.U3)
        uVal1 = torch.transpose(uVal1, 0, 1)
        uVal2 = torch.transpose(uVal2, 0, 1)
        uVal3 = torch.transpose(uVal3, 0, 1)
        uVal1 = uVal1.contiguous().view(batch_size, self.hidden_size)
        uVal2 = uVal2.contiguous().view(batch_size, self.hidden_size)
        uVal3 = uVal3.contiguous().view(batch_size, self.hidden_size)

        #############################
        if self.uRanks[1] > 0:
            h3 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h3 = h3[:, index, :]
            h3 = torch.transpose(h3, 0, 1)
            h3 = torch.bmm(h3, self.UU)
            uuVal1 = torch.bmm(h3, self.UU1)
            uuVal2 = torch.bmm(h3, self.UU2)
            uuVal3 = torch.bmm(h3, self.UU3)
            uuVal1 = torch.transpose(uuVal1, 0, 1)
            uuVal2 = torch.transpose(uuVal2, 0, 1)
            uuVal3 = torch.transpose(uuVal3, 0, 1)
            uuVal1 = uuVal1.contiguous().view(batch_size, self.hidden_size)
            uuVal2 = uuVal2.contiguous().view(batch_size, self.hidden_size)
            uuVal3 = uuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            index = index[1:] + index[0:1]
            uuVal1 = 0
            uuVal2 = 0
            uuVal3 = 0

        #############################
        if self.uRanks[2] > 0:
            h4 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h4 = h4[:, index, :]
            h4 = torch.transpose(h4, 0, 1)
            h4 = torch.bmm(h4, self.UUU)
            uuuVal1 = torch.bmm(h4, self.UUU1)
            uuuVal2 = torch.bmm(h4, self.UUU2)
            uuuVal3 = torch.bmm(h4, self.UUU3)
            uuuVal1 = torch.transpose(uuuVal1, 0, 1)
            uuuVal2 = torch.transpose(uuuVal2, 0, 1)
            uuuVal3 = torch.transpose(uuuVal3, 0, 1)
            uuuVal1 = uuuVal1.contiguous().view(batch_size, self.hidden_size)
            uuuVal2 = uuuVal2.contiguous().view(batch_size, self.hidden_size)
            uuuVal3 = uuuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            uuuVal1 = 0
            uuuVal2 = 0
            uuuVal3 = 0

        #############################
        if self.uRanks[3] > 0:
            h5 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h5 = h5[:, index, :]
            h5 = torch.transpose(h5, 0, 1)
            h5 = torch.bmm(h5, self.UUUU)
            uuuuVal1 = torch.bmm(h5, self.UUUU1)
            uuuuVal2 = torch.bmm(h5, self.UUUU2)
            uuuuVal3 = torch.bmm(h5, self.UUUU3)
            uuuuVal1 = torch.transpose(uuuuVal1, 0, 1)
            uuuuVal2 = torch.transpose(uuuuVal2, 0, 1)
            uuuuVal3 = torch.transpose(uuuuVal3, 0, 1)
            uuuuVal1 = uuuuVal1.contiguous().view(batch_size, self.hidden_size)
            uuuuVal2 = uuuuVal2.contiguous().view(batch_size, self.hidden_size)
            uuuuVal3 = uuuuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            uuuuVal1 = 0
            uuuuVal2 = 0
            uuuuVal3 = 0

        if self.wRank is None:
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
        else:
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)

        matVal_r = wVal1 + uVal1 + uuVal1 + uuuVal1 + uuuuVal1
        matVal_z = wVal2 + uVal2 + uuVal2 + uuuVal2 + uuuuVal2

        r = torch.sigmoid(matVal_r + self.bias_r)
        z = torch.sigmoid(matVal_z + self.bias_gate)

        matVal_c = wVal3 + r * (uVal3 + uuVal3 + uuuVal3 + uuuuVal3)

        c_tilda = torch.tanh(matVal_c + self.bias_update)

        h_next = z * h + (1.0 - z) * c_tilda

        return h_next


class myGRUCell_group5(nn.Module):
    """
    wRank = rank of W matrix
    (creates 4 matrices if not None else creates 3 matrices)
    uRank = rank of U matrix
    (creates 4 matrices if not None else creates 3 matrices)

    Basic architecture is like:

    r_t = gate_nl(W1x_t + U1h_{t-1} + B_r)
    z_t = gate_nl(W2x_t + U2h_{t-1} + B_g)
    h_t^ = update_nl(W3x_t + r_t*U3(h_{t-1}) + B_h)
    h_t = z_t*h_{t-1} + (1-z_t)*h_t^

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    """

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, g=5, recurrent_init=None,
                 hidden_init=None):
        super(myGRUCell_group5, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
        self.g = g
        #print("wRank is:{}".format(wRank))
        #print("uRank is:{}".format(uRanks))

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRanks is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            # self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            # self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U1_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            # self.U2_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            # self.U3_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            self.U = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[0]]))
            self.U1 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U2 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U3 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))

            self.UU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[1]]))
            self.UU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))

            if uRanks[2] > 0:
                self.UUU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[2]]))
                self.UUU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))
                self.UUU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))
                self.UUU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))
            if uRanks[3] > 0:
                self.UUUU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[3]]))
                self.UUUU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[3], int(hidden_size / g)]))
                self.UUUU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[3], int(hidden_size / g)]))
                self.UUUU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[3], int(hidden_size / g)]))
            if uRanks[4] > 0:
                self.UUUUU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[4]]))
                self.UUUUU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[4], int(hidden_size / g)]))
                self.UUUUU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[4], int(hidden_size / g)]))
                self.UUUUU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[4], int(hidden_size / g)]))

        self.bias_r = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, h):

        index = list(range(self.g))

        #############################
        batch_size = h.shape[0]
        h2 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
        h2 = torch.transpose(h2, 0, 1)
        h2 = torch.bmm(h2, self.U)
        uVal1 = torch.bmm(h2, self.U1)
        uVal2 = torch.bmm(h2, self.U2)
        uVal3 = torch.bmm(h2, self.U3)
        uVal1 = torch.transpose(uVal1, 0, 1)
        uVal2 = torch.transpose(uVal2, 0, 1)
        uVal3 = torch.transpose(uVal3, 0, 1)
        uVal1 = uVal1.contiguous().view(batch_size, self.hidden_size)
        uVal2 = uVal2.contiguous().view(batch_size, self.hidden_size)
        uVal3 = uVal3.contiguous().view(batch_size, self.hidden_size)

        #############################
        if self.uRanks[1] > 0:
            h3 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h3 = h3[:, index, :]
            h3 = torch.transpose(h3, 0, 1)
            h3 = torch.bmm(h3, self.UU)
            uuVal1 = torch.bmm(h3, self.UU1)
            uuVal2 = torch.bmm(h3, self.UU2)
            uuVal3 = torch.bmm(h3, self.UU3)
            uuVal1 = torch.transpose(uuVal1, 0, 1)
            uuVal2 = torch.transpose(uuVal2, 0, 1)
            uuVal3 = torch.transpose(uuVal3, 0, 1)
            uuVal1 = uuVal1.contiguous().view(batch_size, self.hidden_size)
            uuVal2 = uuVal2.contiguous().view(batch_size, self.hidden_size)
            uuVal3 = uuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            index = index[1:] + index[0:1]
            uuVal1 = 0
            uuVal2 = 0
            uuVal3 = 0

        #############################
        if self.uRanks[2] > 0:
            h4 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h4 = h4[:, index, :]
            h4 = torch.transpose(h4, 0, 1)
            h4 = torch.bmm(h4, self.UUU)
            uuuVal1 = torch.bmm(h4, self.UUU1)
            uuuVal2 = torch.bmm(h4, self.UUU2)
            uuuVal3 = torch.bmm(h4, self.UUU3)
            uuuVal1 = torch.transpose(uuuVal1, 0, 1)
            uuuVal2 = torch.transpose(uuuVal2, 0, 1)
            uuuVal3 = torch.transpose(uuuVal3, 0, 1)
            uuuVal1 = uuuVal1.contiguous().view(batch_size, self.hidden_size)
            uuuVal2 = uuuVal2.contiguous().view(batch_size, self.hidden_size)
            uuuVal3 = uuuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            uuuVal1 = 0
            uuuVal2 = 0
            uuuVal3 = 0

        #############################
        if self.uRanks[3] > 0:
            h5 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h5 = h5[:, index, :]
            h5 = torch.transpose(h5, 0, 1)
            h5 = torch.bmm(h5, self.UUUU)
            uuuuVal1 = torch.bmm(h5, self.UUUU1)
            uuuuVal2 = torch.bmm(h5, self.UUUU2)
            uuuuVal3 = torch.bmm(h5, self.UUUU3)
            uuuuVal1 = torch.transpose(uuuuVal1, 0, 1)
            uuuuVal2 = torch.transpose(uuuuVal2, 0, 1)
            uuuuVal3 = torch.transpose(uuuuVal3, 0, 1)
            uuuuVal1 = uuuuVal1.contiguous().view(batch_size, self.hidden_size)
            uuuuVal2 = uuuuVal2.contiguous().view(batch_size, self.hidden_size)
            uuuuVal3 = uuuuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            uuuuVal1 = 0
            uuuuVal2 = 0
            uuuuVal3 = 0

        #############################
        if self.uRanks[4] > 0:
            h6 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h6 = h6[:, index, :]
            h6 = torch.transpose(h6, 0, 1)
            h6 = torch.bmm(h6, self.UUUUU)
            uuuuuVal1 = torch.bmm(h5, self.UUUUU1)
            uuuuuVal2 = torch.bmm(h5, self.UUUUU2)
            uuuuuVal3 = torch.bmm(h5, self.UUUUU3)
            uuuuuVal1 = torch.transpose(uuuuuVal1, 0, 1)
            uuuuuVal2 = torch.transpose(uuuuuVal2, 0, 1)
            uuuuuVal3 = torch.transpose(uuuuuVal3, 0, 1)
            uuuuuVal1 = uuuuuVal1.contiguous().view(batch_size, self.hidden_size)
            uuuuuVal2 = uuuuuVal2.contiguous().view(batch_size, self.hidden_size)
            uuuuuVal3 = uuuuuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            uuuuuVal1 = 0
            uuuuuVal2 = 0
            uuuuuVal3 = 0

        if self.wRank is None:
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
        else:
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)

        matVal_r = wVal1 + uVal1 + uuVal1 + uuuVal1 + uuuuVal1 + uuuuuVal1
        matVal_z = wVal2 + uVal2 + uuVal2 + uuuVal2 + uuuuVal2 + uuuuuVal2

        r = torch.sigmoid(matVal_r + self.bias_r)
        z = torch.sigmoid(matVal_z + self.bias_gate)

        matVal_c = wVal3 + r * (uVal3 + uuVal3 + uuuVal3 + uuuuVal3 + uuuuuVal3)

        c_tilda = torch.tanh(matVal_c + self.bias_update)

        h_next = z * h + (1.0 - z) * c_tilda

        return h_next


class myGRUCell_group6(nn.Module):
    """
    wRank = rank of W matrix
    (creates 4 matrices if not None else creates 3 matrices)
    uRank = rank of U matrix
    (creates 4 matrices if not None else creates 3 matrices)

    Basic architecture is like:

    r_t = gate_nl(W1x_t + U1h_{t-1} + B_r)
    z_t = gate_nl(W2x_t + U2h_{t-1} + B_g)
    h_t^ = update_nl(W3x_t + r_t*U3(h_{t-1}) + B_h)
    h_t = z_t*h_{t-1} + (1-z_t)*h_t^

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    """

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, g=5, recurrent_init=None,
                 hidden_init=None):
        super(myGRUCell_group6, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
        self.g = g
        #print("wRank is:{}".format(wRank))
        #print("uRank is:{}".format(uRanks))

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRanks is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            # self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            # self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            # self.U1_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            # self.U2_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            # self.U3_diag = nn.Parameter(0.1 * torch.randn([hidden_size]))
            self.U = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[0]]))
            self.U1 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U2 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U3 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))

            self.UU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[1]]))
            self.UU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))

            if uRanks[2] > 0:
                self.UUU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[2]]))
                self.UUU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))
                self.UUU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))
                self.UUU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[2], int(hidden_size / g)]))
            if uRanks[3] > 0:
                self.UUUU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[3]]))
                self.UUUU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[3], int(hidden_size / g)]))
                self.UUUU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[3], int(hidden_size / g)]))
                self.UUUU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[3], int(hidden_size / g)]))
            if uRanks[4] > 0:
                self.UUUUU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[4]]))
                self.UUUUU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[4], int(hidden_size / g)]))
                self.UUUUU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[4], int(hidden_size / g)]))
                self.UUUUU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[4], int(hidden_size / g)]))
            if uRanks[5] > 0:
                self.UUUUUU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[5]]))
                self.UUUUUU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[5], int(hidden_size / g)]))
                self.UUUUUU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[5], int(hidden_size / g)]))
                self.UUUUUU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[5], int(hidden_size / g)]))

        self.bias_r = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, h):

        index = list(range(self.g))

        #############################
        batch_size = h.shape[0]
        h2 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
        h2 = torch.transpose(h2, 0, 1)
        h2 = torch.bmm(h2, self.U)
        uVal1 = torch.bmm(h2, self.U1)
        uVal2 = torch.bmm(h2, self.U2)
        uVal3 = torch.bmm(h2, self.U3)
        uVal1 = torch.transpose(uVal1, 0, 1)
        uVal2 = torch.transpose(uVal2, 0, 1)
        uVal3 = torch.transpose(uVal3, 0, 1)
        uVal1 = uVal1.contiguous().view(batch_size, self.hidden_size)
        uVal2 = uVal2.contiguous().view(batch_size, self.hidden_size)
        uVal3 = uVal3.contiguous().view(batch_size, self.hidden_size)

        #############################
        if self.uRanks[1] > 0:
            h3 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h3 = h3[:, index, :]
            h3 = torch.transpose(h3, 0, 1)
            h3 = torch.bmm(h3, self.UU)
            uuVal1 = torch.bmm(h3, self.UU1)
            uuVal2 = torch.bmm(h3, self.UU2)
            uuVal3 = torch.bmm(h3, self.UU3)
            uuVal1 = torch.transpose(uuVal1, 0, 1)
            uuVal2 = torch.transpose(uuVal2, 0, 1)
            uuVal3 = torch.transpose(uuVal3, 0, 1)
            uuVal1 = uuVal1.contiguous().view(batch_size, self.hidden_size)
            uuVal2 = uuVal2.contiguous().view(batch_size, self.hidden_size)
            uuVal3 = uuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            index = index[1:] + index[0:1]
            uuVal1 = 0
            uuVal2 = 0
            uuVal3 = 0

        #############################
        if self.uRanks[2] > 0:
            h4 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h4 = h4[:, index, :]
            h4 = torch.transpose(h4, 0, 1)
            h4 = torch.bmm(h4, self.UUU)
            uuuVal1 = torch.bmm(h4, self.UUU1)
            uuuVal2 = torch.bmm(h4, self.UUU2)
            uuuVal3 = torch.bmm(h4, self.UUU3)
            uuuVal1 = torch.transpose(uuuVal1, 0, 1)
            uuuVal2 = torch.transpose(uuuVal2, 0, 1)
            uuuVal3 = torch.transpose(uuuVal3, 0, 1)
            uuuVal1 = uuuVal1.contiguous().view(batch_size, self.hidden_size)
            uuuVal2 = uuuVal2.contiguous().view(batch_size, self.hidden_size)
            uuuVal3 = uuuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            uuuVal1 = 0
            uuuVal2 = 0
            uuuVal3 = 0

        #############################
        if self.uRanks[3] > 0:
            h5 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h5 = h5[:, index, :]
            h5 = torch.transpose(h5, 0, 1)
            h5 = torch.bmm(h5, self.UUUU)
            uuuuVal1 = torch.bmm(h5, self.UUUU1)
            uuuuVal2 = torch.bmm(h5, self.UUUU2)
            uuuuVal3 = torch.bmm(h5, self.UUUU3)
            uuuuVal1 = torch.transpose(uuuuVal1, 0, 1)
            uuuuVal2 = torch.transpose(uuuuVal2, 0, 1)
            uuuuVal3 = torch.transpose(uuuuVal3, 0, 1)
            uuuuVal1 = uuuuVal1.contiguous().view(batch_size, self.hidden_size)
            uuuuVal2 = uuuuVal2.contiguous().view(batch_size, self.hidden_size)
            uuuuVal3 = uuuuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            uuuuVal1 = 0
            uuuuVal2 = 0
            uuuuVal3 = 0

        #############################
        if self.uRanks[4] > 0:
            h6 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h6 = h6[:, index, :]
            h6 = torch.transpose(h6, 0, 1)
            h6 = torch.bmm(h6, self.UUUUU)
            uuuuuVal1 = torch.bmm(h5, self.UUUUU1)
            uuuuuVal2 = torch.bmm(h5, self.UUUUU2)
            uuuuuVal3 = torch.bmm(h5, self.UUUUU3)
            uuuuuVal1 = torch.transpose(uuuuuVal1, 0, 1)
            uuuuuVal2 = torch.transpose(uuuuuVal2, 0, 1)
            uuuuuVal3 = torch.transpose(uuuuuVal3, 0, 1)
            uuuuuVal1 = uuuuuVal1.contiguous().view(batch_size, self.hidden_size)
            uuuuuVal2 = uuuuuVal2.contiguous().view(batch_size, self.hidden_size)
            uuuuuVal3 = uuuuuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            uuuuuVal1 = 0
            uuuuuVal2 = 0
            uuuuuVal3 = 0

        #############################
        if self.uRanks[5] > 0:
            h7 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h7 = h7[:, index, :]
            h7 = torch.transpose(h7, 0, 1)
            h7 = torch.bmm(h7, self.UUUUUU)
            uuuuuuVal1 = torch.bmm(h5, self.UUUUUU1)
            uuuuuuVal2 = torch.bmm(h5, self.UUUUUU2)
            uuuuuuVal3 = torch.bmm(h5, self.UUUUUU3)
            uuuuuuVal1 = torch.transpose(uuuuuuVal1, 0, 1)
            uuuuuuVal2 = torch.transpose(uuuuuuVal2, 0, 1)
            uuuuuuVal3 = torch.transpose(uuuuuuVal3, 0, 1)
            uuuuuuVal1 = uuuuuuVal1.contiguous().view(batch_size, self.hidden_size)
            uuuuuuVal2 = uuuuuuVal2.contiguous().view(batch_size, self.hidden_size)
            uuuuuuVal3 = uuuuuuVal3.contiguous().view(batch_size, self.hidden_size)
        else:
            uuuuuuVal1 = 0
            uuuuuuVal2 = 0
            uuuuuuVal3 = 0

        if self.wRank is None:
            wVal1 = torch.matmul(x, self.W1)
            wVal2 = torch.matmul(x, self.W2)
            wVal3 = torch.matmul(x, self.W3)
        else:
            wVal1 = torch.matmul(
                torch.matmul(x, self.W), self.W1)
            wVal2 = torch.matmul(
                torch.matmul(x, self.W), self.W2)
            wVal3 = torch.matmul(
                torch.matmul(x, self.W), self.W3)

        matVal_r = wVal1 + uVal1 + uuVal1 + uuuVal1 + uuuuVal1 + uuuuuVal1 + uuuuuuVal1
        matVal_z = wVal2 + uVal2 + uuVal2 + uuuVal2 + uuuuVal2 + uuuuuVal2 + uuuuuuVal2

        r = torch.sigmoid(matVal_r + self.bias_r)
        z = torch.sigmoid(matVal_z + self.bias_gate)

        matVal_c = wVal3 + r * (uVal3 + uuVal3 + uuuVal3 + uuuuVal3 + uuuuuVal3 + uuuuuuVal3)

        c_tilda = torch.tanh(matVal_c + self.bias_update)

        h_next = z * h + (1.0 - z) * c_tilda

        return h_next

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

    def __init__(self, input_size, hidden_size, wRank=None, uRank=None, recurrent_init=None,
                 hidden_init=None):
        super(myLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRank = uRank

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

        if uRank is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U4 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U4 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))

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

        if self.uRank is None:
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
        return c_next, h_next


class myLSTMCell_group2(nn.Module):
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

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, g=2, recurrent_init=None,
                 hidden_init=None):
        super(myLSTMCell_group2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
        self.g = g

        #print("wRank is:{}".format(wRank))
        #print("uRank is:{}".format(uRanks))

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
            self.U = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[0]]))
            self.U1 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U2 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U3 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))
            self.U4 = nn.Parameter(0.1 * torch.randn([g, uRanks[0], int(hidden_size / g)]))

            self.UU = nn.Parameter(0.1 * torch.randn([g, int(hidden_size / g), uRanks[1]]))
            self.UU1 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU2 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU3 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))
            self.UU4 = nn.Parameter(0.1 * torch.randn([g, uRanks[1], int(hidden_size / g)]))

        self.bias_f = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_i = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_c = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_o = nn.Parameter(torch.ones([1, hidden_size]))

    def forward(self, x, hiddenStates):
        (h, c) = hiddenStates
        index = list(range(self.g))

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
            #############################
            batch_size = h.shape[0]
            h2 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            h2 = torch.transpose(h2, 0, 1)
            h2 = torch.bmm(h2, self.U)
            uVal1 = torch.bmm(h2, self.U1)
            uVal2 = torch.bmm(h2, self.U2)
            uVal3 = torch.bmm(h2, self.U3)
            uVal4 = torch.bmm(h2, self.U4)
            uVal1 = torch.transpose(uVal1, 0, 1)
            uVal2 = torch.transpose(uVal2, 0, 1)
            uVal3 = torch.transpose(uVal3, 0, 1)
            uVal4 = torch.transpose(uVal4, 0, 1)
            uVal1 = uVal1.contiguous().view(batch_size, self.hidden_size)
            uVal2 = uVal2.contiguous().view(batch_size, self.hidden_size)
            uVal3 = uVal3.contiguous().view(batch_size, self.hidden_size)
            uVal4 = uVal4.contiguous().view(batch_size, self.hidden_size)

            #############################
            h3 = h.view(batch_size, self.g, int(self.hidden_size / self.g))
            index = index[1:] + index[0:1]
            h3 = h3[:, index, :]
            h3 = torch.transpose(h3, 0, 1)
            h3 = torch.bmm(h3, self.UU)
            uuVal1 = torch.bmm(h3, self.UU1)
            uuVal2 = torch.bmm(h3, self.UU2)
            uuVal3 = torch.bmm(h3, self.UU3)
            uuVal4 = torch.bmm(h3, self.UU4)
            uuVal1 = torch.transpose(uuVal1, 0, 1)
            uuVal2 = torch.transpose(uuVal2, 0, 1)
            uuVal3 = torch.transpose(uuVal3, 0, 1)
            uuVal4 = torch.transpose(uuVal4, 0, 1)
            uuVal1 = uuVal1.contiguous().view(batch_size, self.hidden_size)
            uuVal2 = uuVal2.contiguous().view(batch_size, self.hidden_size)
            uuVal3 = uuVal3.contiguous().view(batch_size, self.hidden_size)
            uuVal4 = uuVal4.contiguous().view(batch_size, self.hidden_size)

        matVal_i = wVal1 + uVal1 + uuVal1
        matVal_f = wVal2 + uVal2 + uuVal2
        matVal_o = wVal3 + uVal3 + uuVal3
        matVal_c = wVal4 + uVal4 + uuVal4

        i = torch.sigmoid(matVal_i + self.bias_i)
        f = torch.sigmoid(matVal_f + self.bias_f)
        o = torch.sigmoid(matVal_o + self.bias_o)

        c_tilda = torch.tanh(matVal_c + self.bias_c)

        c_next = f * c + i * c_tilda
        h_next = o * torch.tanh(c_next)
        return c_next, h_next

# module which have multiple layer of myGRU cell
class myGRU(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRank=None, **kwargs):
        super(myGRU, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRank = uRank
        print("start training with wRank:{}".format(wRank))
        print("start training with uRank:{}".format(uRank))

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
            rnn_cells.append(myGRUCell(in_size, hidden_size, wRank=self.wRank, uRank=self.uRank, **kwargs))
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
            hx = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                hx = cell(x_time[t], hx)
                outputs.append(hx)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(hx)
            i = i + 1

        return x, torch.cat(hiddens, -1)

# module which have multiple layer of myGRU_group cell
class myGRU_group(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank_diag=None, wRank_offdiag=None, uRank_diag=None, uRank_offdiag=None, **kwargs):
        super(myGRU_group, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank_diag = wRank_diag
        self.uRank_diag = uRank_diag
        self.wRank_offdiag = wRank_offdiag
        self.uRank_offdiag = uRank_offdiag
        print("wRank_diag is:{}".format(wRank_diag))
        print("wRank_offdiag is:{}".format(wRank_offdiag))
        print("uRank_diag is:{}".format(uRank_diag))
        print("uRank_offdiag is:{}".format(uRank_offdiag))

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
            rnn_cells.append(
                myGroupGRUCell(in_size, hidden_size, wRank_diag=self.wRank_diag, wRank_offdiag=self.wRank_offdiag,
                               uRank_diag=self.uRank_diag, uRank_offdiag=self.uRank_offdiag, **kwargs))
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
            hx = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                hx = cell(x_time[t], hx)
                outputs.append(hx)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(hx)
            i = i + 1

        return x, torch.cat(hiddens, -1)

# module which have multiple layer of myGRU_group2 cell
class myGRU_group2(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRanks=None, **kwargs):
        super(myGRU_group2, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRanks = uRanks
        print("start training with wRank:{}".format(wRank))
        print("start training with uRank:{}".format(uRanks))

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
            rnn_cells.append(myGRUCell_group2(in_size, hidden_size, wRank=self.wRank, uRanks=self.uRanks, **kwargs))
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
            hx = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                hx = cell(x_time[t], hx)
                outputs.append(hx)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(hx)
            i = i + 1

        return x, torch.cat(hiddens, -1)

# module which have multiple layer of myGRU_group3 cell
class myGRU_group3(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRanks=None, **kwargs):
        super(myGRU_group3, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRanks = uRanks
        print("start training with wRank:{}".format(wRank))
        print("start training with uRank:{}".format(uRanks))

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
            rnn_cells.append(myGRUCell_group3(in_size, hidden_size, wRank=self.wRank, uRanks=self.uRanks, **kwargs))
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
            hx = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                hx = cell(x_time[t], hx)
                outputs.append(hx)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(hx)
            i = i + 1

        return x, torch.cat(hiddens, -1)

# module which have multiple layer of myGRU_group4 cell
class myGRU_group4(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRanks=None, **kwargs):
        super(myGRU_group4, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRanks = uRanks
        print("start training with wRank:{}".format(wRank))
        print("start training with uRank:{}".format(uRanks))

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
            rnn_cells.append(myGRUCell_group4(in_size, hidden_size, wRank=self.wRank, uRanks=self.uRanks, **kwargs))
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
            hx = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                hx = cell(x_time[t], hx)
                outputs.append(hx)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(hx)
            i = i + 1

        return x, torch.cat(hiddens, -1)

# module which have multiple layer of myGRU_group5 cell
class myGRU_group5(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRanks=None, **kwargs):
        super(myGRU_group5, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRanks = uRanks
        print("start training with wRank:{}".format(wRank))
        print("start training with uRank:{}".format(uRanks))

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
            rnn_cells.append(myGRUCell_group5(in_size, hidden_size, wRank=self.wRank, uRanks=self.uRanks, **kwargs))
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
            hx = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                hx = cell(x_time[t], hx)
                outputs.append(hx)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(hx)
            i = i + 1

        return x, torch.cat(hiddens, -1)

# module which have multiple layer of myGRU_group6 cell
class myGRU_group6(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRanks=None, **kwargs):
        super(myGRU_group6, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        self.wRank = wRank
        self.uRanks = uRanks
        print("start training with wRank:{}".format(wRank))
        print("start training with uRank:{}".format(uRanks))

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
            rnn_cells.append(myGRUCell_group6(in_size, hidden_size, wRank=self.wRank, uRanks=self.uRanks, **kwargs))
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
            hx = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                hx = cell(x_time[t], hx)
                outputs.append(hx)
            x = torch.stack(outputs, time_index)
            # x=torch.cat(outputs, -1)
            hiddens.append(hx)
            i = i + 1

        return x, torch.cat(hiddens, -1)


# Code for implementing DeepConvLSTM
class DeepConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRank=None, **kwargs):
        super(DeepConvLSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 1))
        self.conv2 = nn.Conv2d(64, 64, (5, 1))
        self.conv3 = nn.Conv2d(64, 64, (5, 1))
        self.conv4 = nn.Conv2d(64, 64, (5, 1))

        # self.lstm1 = nn.LSTM(7232, 128, batch_first = True)
        # self.lstm2 = nn.LSTM(128, 128, batch_first = True)
        self.lstm = myLSTM(7232, hidden_layer_sizes=[128, 128], batch_first=True)
        self.gru = myGRU(7232, hidden_layer_sizes=[128, 128], batch_first=True)
        # self.gru1 = nn.LSTM(7232, 128)
        # self.gru2 = nn.LSTM(128, 128)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, hidden=None):
        self.device = x.device
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))

        x, h = self.gru(x)

        """
        h0 = torch.zeros(1, x.size(0), 128).to(self.device)
        c0 = torch.zeros(1, x.size(0), 128).to(self.device)

        #print(x.shape)
        output, (h, c) = self.lstm1(x, (h0, c0))
        #print(output.shape)

        h1 = torch.zeros(1, output.size(0), 128).to(self.device)
        c1 = torch.zeros(1, output.size(0), 128).to(self.device)

        output, (h, c) = self.lstm2(output, (h1, c1))
        #output = output.permute(1,0,2)
        #output = output[0,:,:]
        """
        #########################################
        return x, h

# Code for implementing DeepConvLSTM
# This is implementation of DeepcConvolutional part, and LSTM part will be added
class DeepConv(nn.Module):
    def __init__(self, filter_size=5, filter_count=64):
        super(DeepConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 1))
        self.conv2 = nn.Conv2d(64, 64, (5, 1))
        self.conv3 = nn.Conv2d(64, 64, (5, 1))
        self.conv4 = nn.Conv2d(64, 64, (5, 1))

        # self.lstm1 = nn.LSTM(7232, 128, batch_first = True)
        # self.lstm2 = nn.LSTM(128, 128, batch_first = True)
        # self.lstm = myLSTM(7232, hidden_layer_sizes=[128, 128], batch_first = True)
        # self.gru = myGRU(7232, hidden_layer_sizes=[128, 128], batch_first = True)
        # self.gru1 = nn.LSTM(7232, 128)
        # self.gru2 = nn.LSTM(128, 128)

    def forward(self, x, hidden=None):
        self.device = x.device
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))

        return x
