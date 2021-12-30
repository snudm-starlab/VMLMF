################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
# Project: Starlab
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: test.py
# - test file for test VMLMF in Human Activity Recognition
#
# Version : 1.0
# Date : Oct 14, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
# pylint: disable=C0103, E1101, C0114, R0902,C0116, R0914, R0913, C0123, W0613, W0102,C0413, E0401
"""
====================================
 :mod:`test`
====================================
.. moduleauthor:: Hyojin Jeon  <tarahjjeon@snu.ac.kr>
설명
=====
학습한 모델의 테스트를 위한 모듈입니다.

"""
import torch
def test(model,test_data,cuda,device):
    """test trained model with test data
 
    :param model: model to test
    :param test_data: test_data loader
    :param args: arguments user decided
    :param cuda: whether cuda is availabel
    :param device: device user uses
    :returns: trained model
    """
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_data:
            if cuda:
                data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print(f"Test accuracy:: {100. * correct / len(test_data.dataset):.4f}")
