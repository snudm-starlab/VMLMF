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

import sys
sys.path.append('./')

from models.compressed_lstm import *
from utils.compression_cal import *
from utils.save_load import *
from utils.OPP_dataloader import *
from utils.UCI_dataloader import *

from train_test.train import train
from train_test.test import test

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from time import time

parser = argparse.ArgumentParser(description='PyTorch group GRU, LSTM testing')
parser.add_argument('--lr', type=float, default=0.002,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-batch-norm', action='store_true', default=False,
                    help='disable frame-wise batch normalization after each layer')
parser.add_argument('--log_epoch', type=int, default=1,
                    help='after how many epochs to report performance')
parser.add_argument('--log_iteration', type=int, default=-1,
                    help='after how many iterations to report performance, deactivates with -1 (default: -1)')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='enable bidirectional processing')
parser.add_argument('--batch-size', type=int, default=81,
                    help='input batch size for training (default: 64)')
parser.add_argument('--max-steps', type=int, default=10000,
                    help='max iterations of training (default: 10000)')
parser.add_argument('--model', type=str, default="myGRU",
                    help='if either myGRU or myLSTM cells should be used for optimization')
parser.add_argument('--layer_sizes', type=int, nargs='+', default=None,
                    help='list of layers')
parser.add_argument('--wRank', type=int, default=None,
                    help='compress rank of non-recurrent weight')
parser.add_argument('--uRanks', type=int, default=None,
                    help='compress rank of recurrent weight')
parser.add_argument('--gpu_id', type=int, default=3,
                    help='gpu_id assign')
parser.add_argument("-train", "--is_train",
                        help="whether train_test the model (train_test-True; test-False)",
                        action="store_true")
parser.add_argument("--seed",type=int,default=3,help='seed')
parser.add_argument("--data",type=str,default="OPP",help='choose dataset (OPP or UCI)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.batch_norm = not args.no_batch_norm

TIME_STEPS = 128
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
RECURRENT_MIN = pow(1 / 2, 1 / TIME_STEPS)

# Code for setting random seed for reproduce
# You can change seed number by setting seed = n
cuda = torch.cuda.is_available()
seed = args.seed

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.autograd.set_detect_anomaly(True)


def main():

    gpu_id = args.gpu_id
    device = 'cuda:{}'.format(gpu_id)
    #print("device",device)

    input_size=77 if args.data.lower()=="opp" else 9
    if args.model.lower()=="vmmodel_neo":
        model = Net(input_size, layer_sizes=args.layer_sizes, wRank=args.wRank, uRanks=args.uRanks,
                model=myLSTM,cell=myVMLSTMCell_NEO3) 
    elif args.model.lower()=="vmmodel":
        model = Net(input_size, layer_sizes=args.layer_sizes, wRank=args.wRank, uRanks=args.uRanks,
                model=myLSTM,cell=myVMLSTMCell_NEO3) 
    elif args.model.lower()=="vmmodel_test":
        model = Net(input_size, layer_sizes=args.layer_sizes, wRank=args.wRank, uRanks=args.uRanks,
                model=myLSTM,cell=myVMLSTMCell_NEO4) 
    elif args.model.lower()=="vmmodel_neo_unrevised":
        model = Net(input_size, layer_sizes=args.layer_sizes, wRank=args.wRank, uRanks=args.uRanks,
                model=myLSTM,cell=myVMLSTMCell_NEO5) 
    elif args.model.lower() =="mylstm":
        model = Net(input_size, layer_sizes=args.layer_sizes,model=myLSTM,cell=myLSTMCell) 
    else:
        raise Exception("unsupported cell model")
    
    if cuda:
        #print(">>> cuda is available\n")
        model.to(device)
    
    # load data
    train_data, test_data = HAR_dataloader(args.batch_size) if args.data.lower() == "opp" else UCI_dataloader(args.batch_size)

    if args.is_train == True:
        trained_model=train(model,train_data,args,cuda,device)
        save_model(trained_model,args) if args.model.lower()!="mylstm" else save_model(trained_model,args,name="vanilla_lstm_layer_{}_seed{}".format(args.layer_sizes,args.seed))

        print("Baseline Model")
        stdmodel=Net(input_size, layer_sizes=args.layer_sizes,
                model=myLSTM,cell=myLSTMCell)
        print_model_parm_nums(stdmodel)
        print_model_parm_flops(stdmodel,len(train_data),args,modeltype="mylstm")

        if args.model.lower() != "mylstm":
            print("Compressed Model")         
            print_model_parm_nums(model)
            print_model_parm_flops(model,len(train_data),args)

    else:
        name="vanilla_lstm_layer_{}_seed{}".format(args.layer_sizes,args.seed) if args.model.lower() == "mylstm" else None
        model=load_model(model,args,name=name)
        test(model,test_data,cuda,device)
        print_model_parm_nums(model)
        print_model_parm_flops(model,len(test_data),args)

if __name__ == "__main__":
    main()
