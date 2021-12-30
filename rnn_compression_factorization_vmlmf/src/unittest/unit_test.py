################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
# Project: Starlab
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: unit_test.py
# - unit_test file for unit_test
#
# Version : 1.0
# Date : Dec 28, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
# pylint: disable=C0103, E1101, C0114, R0902,C0116, R0914, R0913, C0123, W0613, W0102,C0413, E0401
"""
====================================
 :mod:`unit_test`
====================================
.. moduleauthor:: Hyojin Jeon  <tarahjjeon@snu.ac.kr>
설명
=====
단위 테스트를 위한모듈입니다.

"""
import sys
sys.path.append('./')
import unittest
import random
import torch
import numpy as np
from models.vmlmf import MyLSTM, MyVMLMFCell,Net
from models.vmlmf_group import MyVMLMFCellg2

# Fix random seed for reproducibility
def set_seed(seed = 3):
    """set random seed for reproducability"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# problem setting
in_sz=77
hid_sz=180
i2i_rank=8
h2h_rank=6
h2h_g_ranks=[2,4]

x=torch.randn([81,24,77])

#model
vmmodelc=Net(77, layer_sizes=[180], wRank=8, uRanks=6, model=MyLSTM,cell=MyVMLMFCell)
vmmodelg=Net(77, layer_sizes=[180], wRank=8, uRanks=[2, 4], model=MyLSTM,cell=MyVMLMFCellg2)

class TestVMLMF(unittest.TestCase):
    """Unit test class"""
    def test01_mmfc_dia_vec_size(self):
        """Unit test for test diagonal vector size"""
        self.assertEqual(vmmodelc.cell.dia_x.shape,(1,77))  # add assertion here
        self.assertEqual(vmmodelc.cell.dia_x.shape, (1, 180))
    def test02_mmfc_weight_shape(self):
        """Unit test for test weight shape"""
        self.assertEqual(vmmodelc.cell.U_x.shape,(in_sz,i2i_rank))  # add assertion here
        self.assertEqual(vmmodelc.cell.U_h.shape,(hid_sz,h2h_rank))
        self.assertEqual(vmmodelc.cell.V_x.shape,(4*hid_sz,i2i_rank))  # add assertion here
        self.assertEqual(vmmodelc.cell.V_h.shape,(4*hid_sz,h2h_rank))
    def test03_mmfc_forward(self):
        """Unit test for test mmf without group structure"""
        computed=vmmodelc.forward(x)
        self.assertEqual(computed.shape,(81,18))  # add assertion here

    def test04_mmfg_dia_vec_size(self):
        """Unit test for test diagonal vector size"""
        self.assertEqual(vmmodelg.cell.layers['dia_x'].shape,(1,77))  # add assertion here
        self.assertEqual(vmmodelg.cell.layers['dia_h'].shape, (1, 180))
    def test05_mmfg_weight_shape(self):
        """Unit test for test weight shape"""
        self.assertEqual(vmmodelg.cell.layers['Ux'].shape,(in_sz,i2i_rank))  # add assertion here
        self.assertEqual(vmmodelg.cell.layers['Uh_0'].shape,(2,int(hid_sz/2),h2h_g_ranks[0]))
        self.assertEqual(vmmodelg.cell.layers['Uh_1'].shape, (2, int(hid_sz / 2), h2h_g_ranks[1]))
        self.assertEqual(vmmodelg.cell.layers['Vx'].shape,(4*hid_sz,i2i_rank))  # add assertion here
        self.assertEqual(vmmodelg.cell.layers['Vh_0'].shape, (2, h2h_g_ranks[0],4*int(hid_sz/2)))
        self.assertEqual(vmmodelg.cell.layers['Vh_1'].shape, (2, h2h_g_ranks[1],4*int(hid_sz/2)))
    def test06_mmfg_forward(self):
        """Unit test for test mmf with group structure"""
        computed = vmmodelc.forward(x)
        self.assertEqual(computed.shape, (81, 18))  # add assertion here

if __name__ == '__main__':
    unittest.main()
