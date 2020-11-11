"""
Module using implemented group GRU, LSTM cell as a building block for
classifying activity from sensor data
"""

from compressed_rnn import myGRU, myLSTM, myGRU2, myLSTM2, DeepConvLSTM, DeepConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from time import time
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='PyTorch deepconv lowrank GRU, LSTM testing')
parser.add_argument('--lr', type=float, default=0.002,
                    help='learning rate (default: 0.002)')
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
                    help='input batch size for training (default: 64)')
parser.add_argument('--max-steps', type=int, default=1,
                    help='max iterations of training (default: 10000)')
parser.add_argument('--model', type=str, default="myGRU",
                    help='if either myGRU or myLSTM cells should be used for optimization')
parser.add_argument('--layer_sizes', type=int, nargs='+', default=None,
                    help='list of layers')
parser.add_argument('--wRank', type=int, default=None,
                    help='compress rank of non-recurrent weight')
parser.add_argument('--uRank', type=int, default=None,
                    help='compress rank of recurrent weight')
parser.add_argument('--gpu', type=int, default=2,
                    help='gpu_id')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.batch_norm = not args.no_batch_norm

# Parameters taken from https://arxiv.org/abs/1803.04831
TIME_STEPS = 24  # 28x28 pixels
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
RECURRENT_MIN = pow(1 / 2, 1 / TIME_STEPS)

cuda = torch.cuda.is_available()
seed = 3

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class Net(nn.Module):
    def __init__(self, input_size, layer_sizes=[32, 32], wRank=None, uRank=None, model=myGRU):
        super(Net, self).__init__()
        recurrent_inits = []
        self.deepconv = DeepConv()

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
            wRank=wRank, uRank=uRank
        )
        self.lin = nn.Linear(layer_sizes[-1], 18)
        # self.lin = nn.Linear(layer_sizes[-1], 10)

        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

    def forward(self, x, hidden=None):
        x = self.deepconv(x)
        y, _ = self.rnn(x, hidden)
        return self.lin(y[:, -1]).squeeze(1)


def main():
    # build model
    if args.model.lower() == "mygru":
        model = Net(7232, layer_sizes=args.layer_sizes, wRank=args.wRank, uRank=args.uRank, model=myGRU)
    elif args.model.lower() == "mylstm":
        model = Net(7232, layer_sizes=args.layer_sizes, wRank=args.wRank, uRank=args.uRank, model=myLSTM)

    else:
        raise Exception("unsupported cell model")
    # model.load_state_dict(torch.load("./weights/{}.pt".format(args.model.lower())))
    # print(model.state_dict())

    device = 'cuda:{}'.format(args.gpu)

    if cuda:
        # model.cuda()
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load data
    train_data, test_data = HAR_dataloader(args.batch_size)

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
                # data, target = data.cuda(), target.cuda()
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
    print(
        "Total Epoch {} cross_entropy {} ({} sec.)".format(
            epochs, np.mean(losses), time() - start))

    # get test error
    model.eval()
    correct = 0
    pred_array = np.array([])
    target_array = np.array([])
    for data_test, target_test in test_data:
        if cuda:
            data_test, target_test = data_test.to(device), target_test.to(device)
        out = model(data_test)
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_test.data.view_as(pred)).cpu().sum()
        pred_array = np.append(pred_array, pred.cpu())
        target_array = np.append(target_array, target_test.cpu())
    print("Test f-score : {:.4f}".format(f1_score(pred_array, target_array, average="macro")))
    #print(f1_score(pred_array, target_array, average=None))
    # print( "Test f-score : {:.4f}".format(100 * f1_score(pred_array, target_array, average=None)))
    print(
        "Test accuracy:: {:.4f}".format(
            100. * correct / len(test_data.dataset)))
    if args.uRank is not None:
        torch.save(model.state_dict(), "./weights/deepconv_lowrank_compressed_{}.pt".format(args.model.lower()))
    else:
        torch.save(model.state_dict(), "./weights/deepconv_full_{}.pt".format(args.model.lower()))


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
                              batch_size=64,
                              shuffle=True, drop_last=True
                              )

    dataset_test = CustomDataset(dataset_folder, 'test')
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=64,
                             shuffle=False
                             )

    return (train_loader, test_loader)


if __name__ == "__main__":
    main()