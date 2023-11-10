################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
# Project: Starlab
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: train.py
# - train file for test VMLMF in Human Activity Recognition
#
# Version : 1.0
# Date : Oct 14, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
# pylint: disable=R0902, R0913, R0914, C0413
"""
====================================
 :mod:`train`
====================================
.. moduleauthor:: Hyojin Jeon  <tarahjjeon@snu.ac.kr>
설명
=====
모델의 학습을 위한 모듈입니다.

"""
from time import time
import torch
import torch.nn.functional as F
import numpy as np


def train(model,train_data,args,cuda,device):
    """train model with train data

    :param model: model to train
    :param train_data: train_data loader
    :param args: arguments user decided
    :param cuda: whether cuda is available or not
    :param device: device users use

    :returns: trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Train the model
    model.train()
    step = 0
    epochs = 0
    start_time = time()
    while epochs < args.max_epochs:
        print(f"****************** EPOCH ={epochs}******************")

        losses = []
        start = time()
        for data, target in train_data:
            if cuda:
                data, target = data.to(device), target.to(device)
            model.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().item())
            step += 1

            if step % args.log_iteration == 0 and args.log_iteration != -1:
                print(f"\tStep {step} cross_entropy {np.mean(losses)}")

        if epochs % args.log_epoch == 0 and args.log_epoch != -1:
            print(f"Epoch {epochs} cross_entropy {np.mean(losses)} ({time() - start} sec.)")
            start = time()
        epochs += 1
    end_time=time()
    print(f'Finished Training. It took {end_time-start_time}s in total')
    return model