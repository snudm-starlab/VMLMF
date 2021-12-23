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
from time import time
import torch
import torch.nn.functional as F
import numpy as np


def train(model,train_data,args,cuda,device):
    """
    train model with train data
    @param model
        model to train
    @param train_data
        train_data loader
    @param args
        arguments user decided
    @param cuda
        whether cuda is availabel
    @param device
        device user uses
    @return trained model
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Train the model
    model.train()
    step = 0
    epochs = 0
    start_time = time()
    while epochs < args.max_epochs:
        print("****************** EPOCH = %d ******************" % epochs)

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
                print(
                    "\tStep {} cross_entropy {}".format(step, np.mean(losses)))
           
        if epochs % args.log_epoch == 0 and args.log_epoch != -1:
            print(
                "Epoch {} cross_entropy {} ({} sec.)".format(
                    epochs, np.mean(losses), time() - start))
            start = time()
        epochs += 1
    end_time=time()
    print('Finished Training. It took %ds in total' % (end_time - start_time))    
    return model
