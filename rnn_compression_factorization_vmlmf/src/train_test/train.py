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
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def train(model,train_data,args,cuda,device):
    
    #if cuda:
    #    print("training with cuda ")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Train the model
    model.train()
    step = 0
    epochs = 0
    start_time = time()
    while step < args.max_steps:
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
                    "\tStep {} cross_entropy {}".format(
                        step, np.mean(losses)))
            if step >= args.max_steps:
                break
        if epochs % args.log_epoch == 0 and args.log_epoch != -1:
            print(
                "Epoch {} cross_entropy {} ({} sec.)".format(
                    epochs, np.mean(losses), time() - start))
            start = time()
        epochs += 1
        #torch.save(model.state_dict(), "./weights/comp_cal_{}.pt".format(args.model.lower()))
    end_time=time()
    print('Finished Training. It took %ds in total' % (end_time - start_time))    
    return model
