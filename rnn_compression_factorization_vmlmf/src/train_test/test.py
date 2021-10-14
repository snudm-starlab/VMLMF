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
import torch
def test(model,test_data,cuda,device):
    """
    test trained model with test data
    @param model
        model to test
    @param test_data
        test_data loader
    @param args
        arguments user decided
    @param cuda
        whether cuda is availabel
    @param device
        device user uses
    @return trained model
    """
    # get test error
    print("test the model")
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in test_data:
            if cuda:
                data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    
    print(
        "Test accuracy:: {:.4f}".format(
            100. * correct / len(test_data.dataset)))
