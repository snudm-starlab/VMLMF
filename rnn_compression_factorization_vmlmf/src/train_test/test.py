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
def test(model,test_data,cuda,device):
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
