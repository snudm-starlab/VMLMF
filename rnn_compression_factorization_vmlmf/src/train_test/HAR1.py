################################################################################
# Starlab RNN-compression with factorization method : Lowrank and group-lowrank rnn
#
# Author: Hyojin Jeon (tarahjjeon@gmail.com), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# Version : 1.0
# Date : Mar 31, 2021
# Main Contact: Donghae Jang
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
from utils.compression_cal2 import *
from utils.earlystopping import EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
import argparse
from time import time



parser = argparse.ArgumentParser(description='PyTorch group GRU, LSTM testing')
parser.add_argument('--lr', type=float, default=0.0251,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-batch-norm', action='store_true', default=False,
                    help='disable frame-wise batch normalization after each layer')
parser.add_argument('--log_epoch', type=int, default=10,
                    help='after how many epochs to report performance')
parser.add_argument('--log_iteration', type=int, default=-1,
                    help='after how many iterations to report performance, deactivates with -1 (default: -1)')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='enable bidirectional processing')
parser.add_argument('--batch-size', type=int, default=100,
                    help='input batch size for training (default: 81)')
parser.add_argument('--valid_bs', type=int, default=100,
                    help='input batch size for training (default: 81)')
parser.add_argument('--test_bs', type=int, default=100,
                    help='input batch size for training (default: 81)')
parser.add_argument('--max-steps', type=int, default=20000,
                    help='max iterations of training (default: 10000)')
parser.add_argument('--max_epochs', type=int, default=120,
                    help='max iterations of training (default: 200)')
parser.add_argument('--model', type=str, default="myGRU",
                    help='if either myGRU or myLSTM cells should be used for optimization')
parser.add_argument('--cell', type=str, default="myLSTMCell",
                    help='if either concate or sum should be used for concatenatation in BidirectionRNN')
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
parser.add_argument('--dataset_folder',type=str,default='./src/data/datavalid',help='choosing datafolder')
parser.add_argument('--factor_epoch',type=int, default=6,help='epoch to start decreasing learning rate')
parser.add_argument('--factor',type=float,default=0,help='if you want to use constant lr, set factor by 0')
parser.add_argument('--dropout',type=float,default=0)
parser.add_argument('--seed',type=int,default=1234)
parser.add_argument('--patience',type=int,default=10)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.batch_norm = not args.no_batch_norm

# Code for setting random seed for reproduce
# You can change seed number by setting seed = n
TIME_STEPS = 81
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
RECURRENT_MIN = pow(1 / 2, 1 / TIME_STEPS)

cuda = torch.cuda.is_available()
seed = args.seed

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# class for making final bidifectional network
class BDNet(nn.Module):
    def __init__(self, input_size, layer_sizes=[179,179], wRank=None, uRanks=None,model=myLSTM,cell_f=myLSTMCell,cell_b=myLSTMCell,concatingmode="concate",winit=0.08,p=0):
        super(BDNet, self).__init__()
        print("0813 revised ")
        self.cell_f = cell_f
        self.cell_b=cell_b
        self.winit=winit
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
            wRank=wRank, uRanks=uRanks,cell=self.cell_f
        )
        self.r_rnn=model(
            input_size, hidden_layer_sizes=layer_sizes,
            batch_first=True, recurrent_inits=recurrent_inits,recurrent=True,
            wRank=wRank, uRanks=uRanks,cell=self.cell_b
        )

        self.drop=nn.Dropout(p=p)
        self.lin= nn.Linear(layer_sizes[-1]*2,18)
        # self.lin = nn.Linear(layer_sizes[-1], 10)

        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.winit, self.winit)
    def forward(self, x, hidden=None):
        y, _ = self.rnn(x, hidden)
        r_x = torch.flip(x,(1,))
        r_y,_ = self.r_rnn(r_x,hidden)

        concated_y=torch.cat((y[:,-1],r_y[:,0]),1)
        concated_y=self.drop(concated_y)
        return self.lin(concated_y).squeeze(1)




def main():
    call_dict = {
        'myLSTM':myLSTM,
        "myLSTMCell":myLSTMCell,
        "myVMLSTM_NEO3":myVMLSTMCell_NEO3,
        "myVMLSTM_final":myVMLSTMCELL_NEO_final
        }
    
    input_size=77 if args.dataset_folder.rsplit("/")[-1] !="data" else 113


    # build model
    if args.model.lower() == "mybilstm":
        model_fullname = "myLSTM" 
        cell_name= "myLSTMCell"
        model = BDNet(input_size, layer_sizes=args.layer_sizes, wRank=args.wRank, uRanks=args.uRanks,concatingmode=args.concatingmode,
                    model=call_dict[model_fullname],cell_f=call_dict[cell_name],cell_b=call_dict[cell_name],p=args.dropout)
    elif args.model.lower()=="vmmodel_final":
        model_fullname = "myLSTM" 
        cell_name="myVMLSTM_final"
        model = BDNet(input_size, layer_sizes=args.layer_sizes,wRank=args.wRank,uRanks=args.uRanks,
                    model=call_dict[model_fullname],cell_f=myVMLSTMCELL_NEO_final,cell_b=myVMLSTMCELL_NEO_final)
    elif args.model.lower()=="vmmodel_hmd":
        model_fullname = "myLSTM" 
        model = BDNet(input_size, layer_sizes=args.layer_sizes,wRank=args.wRank,uRanks=args.uRanks,
                    model=call_dict[model_fullname],cell_f=myHMD,cell_b=myHMD,p=args.dropout)
    else:
        raise Exception("unsupported cell model")

    
    # model.load_state_dict(torch.load("./weights/{}.pt".format(args.model.lower())))
    gpu_id = args.gpu_id
    device = 'cuda:{}'.format(gpu_id)
    print("device",device)

    if cuda:
        print("cuda is available\n")
        model.to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=2)
    # load data
    train_data, valid_data,test_data = HAR_dataloader(batch_size_tr=args.batch_size,batch_size_vd=args.valid_bs,batch_size_tst=args.test_bs, dataset_folder=args.dataset_folder)

    
    earlystopping=EarlyStopping(patience=args.patience,verbose=True)

    # Train the model
    #model.train()
   
    epochs = 0
    start = time()
    learning_rate=args.lr
    #check execution time
    training_time=[]
    while epochs < args.max_epochs:
        model.train()
        train_losses = []
        valid_losses=[]
        #start = time()
        start_training=time()
        if args.factor != 0 and epochs>args.factor_epoch and learning_rate>0.0001:
            learning_rate/=args.factor
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        for data, target in train_data:
            if cuda:
                data, target = data.to(device), target.to(device)
                # data, target = data.cuda(), target.cuda()
            model.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target.long())
            loss.backward()
            optimizer.step()
            #scheduler.step()
            train_losses.append(loss.data.cpu().item())
        
        with torch.no_grad():
            ##validation
            model.eval()
            for data_valid,target_valid in valid_data:
                if cuda:
                    data_valid, target_valid = data_valid.to(device), target_valid.to(device)
                out=model(data_valid)
                loss=F.cross_entropy(out,target_valid.long())
                valid_losses.append(loss.data.cpu().item())


            valid_loss=np.average(valid_losses)
            if epochs % args.log_epoch == 0 and args.log_epoch == -1:
                print( "Epoch {} cross_entropy {} / valid_loss:{} ({} sec.)".format( epochs, np.mean(train_losses), valid_loss,time() - start))    
                start = time()
            
            earlystopping(valid_loss,model)
            if earlystopping.early_stop:
                print("Early Stopping in epoch:{}".format(epochs))
                break
        
        
        epochs += 1
        training_time.append(time()-start_training)
    print("training time average:{}".format(torch.mean(torch.Tensor(training_time))))
    print_model_parm_flops(model,len(train_data),args)
    try:
        model.load_state_dict(torch.load('checkpoint.pt'))
    except:
        pass
    # get test error
    model.eval()
    correct = 0
    pred_array = np.array([])
    target_array = np.array([])
    for data_test, target_test in test_data:
        if cuda:
            #data_test, target_test = data_test.cuda(), target_test.cuda()
            data_test, target_test = data_test.to(device), target_test.to(device)

        out = model(data_test)
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_test.data.view_as(pred)).cpu(). sum()
        pred_array = np.append(pred_array, pred.cpu())
        target_array = np.append(target_array, target_test.cpu())
    """#get # of model params
    print(list(map(id,model.parameters())))"""
    #print("stdmodel info")
    #stdmodelname=model_fullname
    #stdmodel = BDNet(input_size, layer_sizes=args.layer_sizes) if args.model.lower() == "mybilstm" or args.model.lower() =="mybidialstm" else Net(input_size, layer_sizes=args.layer_sizes)
    #stdmodelparams=sum(p.numel() for p in stdmodel.parameters())
    modelparams=sum(p.numel() for p in model.parameters())
    print("{} # of parameters:{}".format(args.model,modelparams))

    print("Test f-score : {:.4f}".format(f1_score(pred_array, target_array, average="macro")))
    print("Test f-score(weighted) : {:.4f}".format(f1_score(pred_array, target_array, average="weighted")))

    #print("Test f-score")
    #print(f1_score(pred_array, target_array, average=None))
    print(
        "Test accuracy:: {:.4f}".format(
            100. * correct / len(test_data.dataset)))
    torch.save(model.state_dict(), "./weights/vmmodel_zero_{}.pt".format(args.model.lower()))


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


def HAR_dataloader(batch_size_tr,batch_size_vd,batch_size_tst, dataset_folder='./src/data/datavalid'):
    dataset_train = CustomDataset(dataset_folder, 'train')
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=batch_size_tr,
                              shuffle=True, drop_last=True
                              )
    dataset_valid = CustomDataset(dataset_folder,'valid')
    valid_loader=DataLoader(dataset=dataset_valid,batch_size=batch_size_vd,shuffle=False,drop_last=True)

    dataset_test = CustomDataset(dataset_folder, 'test')
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=batch_size_tst,
                             shuffle=False
                             )

    return (train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    main()
