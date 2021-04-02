################################################################################
# Starlab RNN-compression with factorization method : Lowrank and group-lowrank rnn
#
# Author: Hyojin Jeon (tarahjjeon@gmail.com), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# Version : 1.0
# Date : Mar 31, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
"""
Module using implemented group GRU, LSTM cell as a building block for
classifying activity from sensor data
"""
from module.compressed_lstm import * 
from compressed_rnn import myGRU, myGRU_group,myGRU_group2,myGRU_group3,myGRU_group4,myGRU_group5,myGRU_group6
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
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
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training (default: 81)')
parser.add_argument('--max-steps', type=int, default=1,
                    help='max iterations of training (default: 10000)')
parser.add_argument('--model', type=str, default="myGRU",
                    help='if either myGRU or myLSTM cells should be used for optimization')
parser.add_argument('--concatingmode', type=str, default="concate",
                    help='if either concate or sum should be used for concatenatation in BidirectionRNN')
parser.add_argument('--layer_sizes', type=int, nargs='+', default=None,
                    help='list of layers')
parser.add_argument('--wRank', type=int, default=None,
                    help='compress rank of non-recurrent weight')
parser.add_argument('--uRanks', type=int, nargs='+', default=None,
                    help='compress rank of recurrent weight')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='gpu_id assign')
parser.add_argument('--group', type=int, default=1,
                    help='choosing # of group')
parser.add_argument('--isShuffle',type=bool, default=False,help='choosing shuffle method')
parser.add_argument('--dataset_folder',type=str,default='./src/data',help='choosing datafolder')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.batch_norm = not args.no_batch_norm

# Code for setting random seed for reproduce
# You can change seed number by setting seed = n
TIME_STEPS = 24
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
RECURRENT_MIN = pow(1 / 2, 1 / TIME_STEPS)

cuda = torch.cuda.is_available()
seed = 3

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# class for making final one-directional network 
class Net(nn.Module):
    def __init__(self, input_size, layer_sizes=[32, 32], wRank=None, uRanks=None, model=myLSTM,cell=myLSTMCell,g=None,isShuffle=False):
        super(Net, self).__init__()
        recurrent_inits = []
        self.cell = cell if g is None or g<2 else myLSTMGroupCell

        n_layer = len(layer_sizes) + 1
        for _ in range(n_layer - 1):
            recurrent_inits.append(
                lambda w: nn.init.uniform_(w, 0, RECURRENT_MAX)
            )

        recurrent_inits.append(lambda w: nn.init.uniform_(
            w, RECURRENT_MIN, RECURRENT_MAX))
        self.rnn = model(
            input_size, hidden_layer_sizes=layer_sizes,
            batch_first=True, recurrent_inits=recurrent_inits,
            wRank=wRank, uRanks=uRanks,cell=self.cell,g=g,isShuffle=isShuffle
        )
        self.lin = nn.Linear(layer_sizes[-1], 18)

        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

    def forward(self, x, hidden=None):
        y, _ = self.rnn(x, hidden)
        return self.lin(y[:, -1]).squeeze(1)

# class for making final bidifectional network
class BDNet(nn.Module):
    def __init__(self, input_size, layer_sizes=[32, 32], wRank=None, uRanks=None,concatingmode="concate", model=myLSTM,cell=myLSTMCell,g=None,isShuffle=False):
        super(BDNet, self).__init__()

        self.cell = cell if g is None or g<2 else myLSTMGroupCell
        recurrent_inits = []

        n_layer = len(layer_sizes) + 1
        for _ in range(n_layer - 1):
            recurrent_inits.append(
                lambda w: nn.init.uniform_(w, 0, RECURRENT_MAX)
            )
        recurrent_inits.append(lambda w: nn.init.uniform_(
            w, RECURRENT_MIN, RECURRENT_MAX))
        self.concatingmode=concatingmode
        self.rnn = model(
            input_size, hidden_layer_sizes=layer_sizes,
            batch_first=True, recurrent_inits=recurrent_inits,recurrent=False,
            wRank=wRank, uRanks=uRanks,g=g,cell=self.cell
        )
        self.r_rnn=model(
            input_size, hidden_layer_sizes=layer_sizes,
            batch_first=True, recurrent_inits=recurrent_inits,recurrent=True,
            wRank=wRank, uRanks=uRanks,g=g,cell=self.cell
        )
       
        self.lin= nn.Linear(layer_sizes[-1]*2,18) if self.concatingmode == "concate" else  nn.Linear(layer_sizes[-1], 18)
        # self.lin = nn.Linear(layer_sizes[-1], 10)

        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

    def forward(self, x, hidden=None):
        y, _ = self.rnn(x, hidden)
        r_x = torch.flip(x,(1,))
        r_y,_ = self.r_rnn(r_x,hidden)
        if self.concatingmode == "concate":
            concated_y=torch.cat((y[:,-1],r_y[:,0]),1)
        elif self.concatingmode == "sum":
            concated_y=torch.add(y[:,-1],r_y[:,0])
        elif self.concatingmode =="avg":
            concated_y=torch.mean((y[:,-1],r_y[:,0]),1) ##avg : concatenate or stack -> todos
        return self.lin(concated_y).squeeze(1)




def main():
    call_dict = {
        'myGRU_group2': myGRU_group2,
        'myGRU_group3': myGRU_group3,
        'myGRU_group4': myGRU_group4,
        'myGRU_group5': myGRU_group5,
        'myGRU_group6': myGRU_group6,
        'myLSTM':myLSTM,
        "myLSTMCell":myLSTMCell,
        "myLSTMGroupCell":myLSTMGroupCell
        }
    
    input_size=77 if args.dataset_folder.rsplit("/")[-1] =="smalldata" else 113


    # build model
    if args.model.lower() == "mygru_group":
        model_fullname = "myGRU_group" + str(args.group)
        model = Net(input_size, layer_sizes=args.layer_sizes, wRank=args.wRank, uRanks=args.uRanks,
                    model=call_dict[model_fullname])
    elif args.model.lower() == "mylstm":
        model_fullname = "myLSTM"
        cell_name="myLSTMGroupCell" if args.group >1 else "myLSTMCell"
        model = Net(input_size, layer_sizes=args.layer_sizes, wRank=args.wRank, uRanks=args.uRanks,
                    model=call_dict[model_fullname],cell=call_dict[cell_name],g=args.group)
    elif args.model.lower() == "mybigru_group":
        model_fullname = "myGRU_group" + str(args.group)
        model = BDNet(input_size, layer_sizes=args.layer_sizes, wRank=args.wRank, uRanks=args.uRanks,concatingmode=args.concatingmode,
                    model=call_dict[model_fullname])
    elif args.model.lower() == "mybilstm":
        model_fullname = "myLSTM" 
        cell_name="myLSTMGroupCell" if args.group >1 else "myLSTMCell"
        model = BDNet(input_size, layer_sizes=args.layer_sizes, wRank=args.wRank, uRanks=args.uRanks,concatingmode=args.concatingmode,
                    model=call_dict[model_fullname],cell=call_dict[cell_name],g=args.group,isShuffle=args.isShuffle)
    else:
        raise Exception("unsupported cell model")
    # model.load_state_dict(torch.load("./weights/{}.pt".format(args.model.lower())))
    gpu_id = args.gpu_id
    device = 'cuda:{}'.format(gpu_id)
    #print(device)

    if cuda:
        print("cuda is available\n")
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load data
    train_data, test_data = HAR_dataloader(batch_size=args.batch_size, dataset_folder=args.dataset_folder)

    # Train the model
    model.train()
    step = 0
    epochs = 0
    start = time()
    while step < args.max_steps:
        losses = []
        #start = time()
        for data, target in train_data:
            if cuda:
                data, target = data.to(device), target.to(device)
                # data, target = data.cuda(), target.cuda()
            model.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().item())
            step += 1

            if step % args.log_iteration == 0 and args.log_iteration != -1:
                print("\tStep {} cross_entropy {}".format(step, np.mean(losses)))
            if step >= args.max_steps:
                break
        if epochs % args.log_epoch == 0 and args.log_epoch != -1:
            print( "Epoch {} cross_entropy {} ({} sec.)".format( epochs, np.mean(losses), time() - start))
            start = time()
        epochs += 1

    # get test error
    model.eval()
    correct = 0
    pred_array = np.array([])
    target_array = np.array([])
    for data_test, target_test in test_data:
        if cuda:
            data_test, target_test = data_test.cuda(), target_test.cuda()
        out = model(data_test)
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_test.data.view_as(pred)).cpu().sum()
        pred_array = np.append(pred_array, pred.cpu())
        target_array = np.append(target_array, target_test.cpu())
    #get # of model params

    stdmodelname=model_fullname
    stdmodel = BDNet(77, layer_sizes=args.layer_sizes, wRank=None, uRanks=None,concatingmode=args.concatingmode, model=call_dict[model_fullname])
    stdmodelparams=sum(p.numel() for p in stdmodel.parameters())
    modelparams=sum(p.numel() for p in model.parameters())
    print("{} (mode: {}) (x{:.4f}) # of parameters:{}".format(args.model,args.concatingmode,stdmodelparams/modelparams,modelparams))

    print("Test f-score : {:.4f}".format(f1_score(pred_array, target_array, average="macro")))
    #print("Test f-score")
    #print(f1_score(pred_array, target_array, average=None))
    print(
        "Test accuracy:: {:.4f}".format(
            100. * correct / len(test_data.dataset)))
    torch.save(model.state_dict(), "./weights/group_lowrank_{}.pt".format(args.model.lower()))


class CustomDataset(Dataset):
    def __init__(self, data_path, mode):
        import numpy as np
        self.X = np.load((data_path + '/' + 'X_' + mode + '.npy'))
        self.y = np.load((data_path + '/' + 'y_' + mode + '.npy'))
        #print(self.X.shape)
        #print(self.y.shape)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


def HAR_dataloader(batch_size, dataset_folder='./src/data'):
    dataset_train = CustomDataset(dataset_folder, 'train')
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True, drop_last=True
                              )

    dataset_test = CustomDataset(dataset_folder, 'test')
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=batch_size,
                             shuffle=False
                             )

    return (train_loader, test_loader)


if __name__ == "__main__":
    main()
