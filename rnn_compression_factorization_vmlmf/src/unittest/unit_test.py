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
# pylint: disable=R0902, R0913, R0914, C0413
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
sys.path.append('../')
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
IN_SZ=77
HID_SZ=180
I2I_RANK=8
H2H_RANK=6
H2H_G_RANKS=[2,4]

x=torch.randn([81,24,77])

#model
vmmodelc=Net(77, layer_sizes=[180], w_rank=8, u_rank=[6], model=MyLSTM,cell=MyVMLMFCell)
vmmodelg=Net(77, layer_sizes=[180], w_rank=8, u_rank=[2, 4], model=MyLSTM,cell=MyVMLMFCellg2)

class TestVMLMF(unittest.TestCase):
    """Unit test class"""
    def test01_mmfc_dia_vec_size(self):
        """Unit test for test diagonal vector size"""
        self.assertEqual(vmmodelc.cell.dia_x.shape,(1,77))  # add assertion here
        self.assertEqual(vmmodelc.cell.dia_h.shape, (1,180))
    def test02_mmfc_weight_shape(self):
        """Unit test for test weight shape"""
        self.assertEqual(vmmodelc.cell.u_x.shape,(IN_SZ,I2I_RANK))  # add assertion here
        self.assertEqual(vmmodelc.cell.u_h.shape,(HID_SZ,H2H_RANK))
        self.assertEqual(vmmodelc.cell.v_x.shape,(4*HID_SZ,I2I_RANK))  # add assertion here
        self.assertEqual(vmmodelc.cell.v_h.shape,(4*HID_SZ,H2H_RANK))
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
        self.assertEqual(vmmodelg.cell.layers['u_x'].shape,(IN_SZ,I2I_RANK))  # add assertion here
        self.assertEqual(vmmodelg.cell.layers['u_h_0'].shape,(2,int(HID_SZ/2),H2H_G_RANKS[0]))
        self.assertEqual(vmmodelg.cell.layers['u_h_1'].shape,(2, int(HID_SZ / 2), H2H_G_RANKS[1]))
        self.assertEqual(vmmodelg.cell.layers['v_x'].shape,(4*HID_SZ,I2I_RANK))
        self.assertEqual(vmmodelg.cell.layers['v_h_0'].shape, (2, H2H_G_RANKS[0],4*int(HID_SZ/2)))
        self.assertEqual(vmmodelg.cell.layers['v_h_1'].shape, (2, H2H_G_RANKS[1],4*int(HID_SZ/2)))
    def test06_mmfg_forward(self):
        """Unit test for test mmf with group structure"""
        computed = vmmodelc.forward(x)
        self.assertEqual(computed.shape, (81, 18))  # add assertion here

if __name__ == '__main__':
    unittest.main()