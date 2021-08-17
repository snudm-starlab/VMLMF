import sys
sys.path.append('./')

import numpy as np
import torch
import torch.nn as nn

import timeit
import argparse
from models.lm_model import Model

#Command line arguments parser. Described as in their 'help' sections.
parser = argparse.ArgumentParser(description="Replication of Zaremba et al. (2014). \n https://arxiv.org/abs/1409.2329")
parser.add_argument("--layer_num", type=int, default=2, help="The number of LSTM layers the model has.")
parser.add_argument("--hidden_size", type=int, default=650, help="The number of hidden units per layer.")
parser.add_argument("--lstm_type", type=str, choices=["pytorch","custom","vmlmf"], default="pytorch", help="Which implementation of LSTM to use."
                    + "Note that 'pytorch' is about 2 times faster.")
parser.add_argument("--dropout", type=float, default=0.5, help="The dropout parameter.")
parser.add_argument("--winit", type=float, default=0.05, help="The weight initialization parameter.")
parser.add_argument("--batch_size", type=int, default=20, help="The batch size.")
parser.add_argument("--seq_length", type=int, default=35, help="The sequence length for bptt.")
parser.add_argument("--learning_rate", type=float, default=1, help="The learning rate.")
parser.add_argument("--total_epochs", type=int, default=39, help="Total number of epochs for training.")
parser.add_argument("--factor_epoch", type=int, default=6, help="The epoch to start factoring the learning rate.")
parser.add_argument("--factor", type=float, default=1.2, help="The factor to decrease the learning rate.")
parser.add_argument("--max_grad_norm", type=float, default=5, help="The maximum norm of gradients we impose on training.")
parser.add_argument("--device", type=str, choices = ["cpu", "gpu"], default = "gpu", help = "Whether to use cpu or gpu."
                    + "On default falls back to gpu if one exists, falls back to cpu otherwise.")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu_id")
parser.add_argument("--wRank", type=int, default=300, help="wRank of vmlmf.")
parser.add_argument("--uRanks", type=int, default=300, help="uRank of vmlmf.")
args = parser.parse_args()

def setdevice():
    if args.device == "gpu" and torch.cuda.is_available():
        print("Model will be training on the GPU.\n")
        args.device = torch.device(f'cuda:{args.gpu_id}')
    elif args.device == "gpu":
        print("No GPU detected. Falling back to CPU.\n")
        args.device = torch.device('cpu')
    else:
        print("Model will be training on the CPU.\n")
        args.device = torch.device('cpu')

setdevice()
print('Parameters of the model:')
print('Args:', args)
print("\n")

def data_init():
    with open("./data/ptb.train.txt") as f:
        file = f.read()
        trn = file[1:].split(' ')
    with open("./data/ptb.valid.txt") as f:
        file = f.read()
        vld = file[1:].split(' ')
    with open("./data/ptb.test.txt") as f:
        file = f.read()
        tst = file[1:].split(' ')
    words = sorted(set(trn))
    char2ind = {c: i for i, c in enumerate(words)}
    trn = [char2ind[c] for c in trn]
    vld = [char2ind[c] for c in vld]
    tst = [char2ind[c] for c in tst]
    return np.array(trn).reshape(-1, 1), np.array(vld).reshape(-1, 1), np.array(tst).reshape(-1, 1), len(words)

#Batches the data with [T, B] dimensionality.
def minibatch(data, batch_size, seq_length):
    data = torch.tensor(data, dtype = torch.int64)
    num_batches = data.size(0)//batch_size
    data = data[:num_batches*batch_size]
    data=data.view(batch_size,-1)
    dataset = []
    for i in range(0,data.size(1)-1,seq_length):
        seqlen=int(np.min([seq_length,data.size(1)-1-i]))
        if seqlen<data.size(1)-1-i:
            x=data[:,i:i+seqlen].transpose(1, 0)
            y=data[:,i+1:i+seqlen+1].transpose(1, 0)
            dataset.append((x, y))
    return dataset

#The loss function.
def nll_loss(scores, y):
    batch_size = y.size(1)
    expscores = scores.exp()
    probabilities = expscores / expscores.sum(1, keepdim = True)
    answerprobs = probabilities[range(len(y.reshape(-1))), y.reshape(-1)]
    #I multiply by batch_size as in the original paper
    #Zaremba et al. sum the loss over batches but average these over time.
    return torch.mean(-torch.log(answerprobs) * batch_size)

def perplexity(data, model):
    with torch.no_grad():
        losses = []
        states = model.state_init(args.batch_size)
        for x, y in data:
            scores, states = model(x, states)
            loss = nll_loss(scores, y)
            #Again with the sum/average implementation described in 'nll_loss'.
            losses.append(loss.data.item()/args.batch_size)
    return np.exp(np.mean(losses))

def train(data, model, epochs, epoch_threshold, lr, factor, max_norm):
    trn, vld, tst = data
    tic = timeit.default_timer()
    total_words = 0
    print("Starting training.\n")
    for epoch in range(epochs):
        states = model.state_init(args.batch_size)
        model.train()
        if epoch > epoch_threshold and lr>0.001:
            lr = lr / factor
        for i, (x, y) in enumerate(trn):
            total_words += x.numel()
            model.zero_grad()
            states = model.detach(states)
            scores, states = model(x, states)
            loss = nll_loss(scores, y)
            loss.backward()
            with torch.no_grad():
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                for name,param in model.named_parameters():
                    try:
                        param -= lr * param.grad
                    except:
                        print(f"param: {name}")
            if i % (len(trn)//10) == 0:
                toc = timeit.default_timer()
                print("batch no = {:d} / {:d}, ".format(i, len(trn)) +
                      "train loss = {:.3f}, ".format(loss.item()/args.batch_size) +
                      "wps = {:d}, ".format(round(total_words/(toc-tic))) +
                      "dw.norm() = {:.3f}, ".format(norm) +
                      "lr = {:.3f}, ".format(lr) +
                      "since beginning = {:d} mins, ".format(round((toc-tic)/60)) + 
                      "cuda memory = {:.3f} GBs".format(torch.cuda.max_memory_allocated()/1024/1024/1024))
        model.eval()
        val_perp = perplexity(vld, model)
        print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch+1, val_perp))
        print("*************************************************\n")
    tst_perp = perplexity(tst, model)
    print("Test set perplexity : {:.3f}".format(tst_perp))
    print("Training is over.")
    
def main():
    trn, vld, tst, vocab_size = data_init()
    trn = minibatch(trn, args.batch_size, args.seq_length)
    vld = minibatch(vld, args.batch_size, args.seq_length)
    tst = minibatch(tst, args.batch_size, args.seq_length)
    model = Model(vocab_size, args.hidden_size, args.layer_num, args.dropout, args.winit,args.wRank,args.uRanks, args.lstm_type,device=args.device)
    print("*"*32)
    params=sum(p.numel() for p in model.parameters())
    print("*pameters of model{}: {}M".format(args.lstm_type,params/1e6))
    #print("Compression rate:x{}".format(19780400/params))
    print("*"*32)
    model.to(args.device)
    train((trn, vld, tst), model, args.total_epochs, args.factor_epoch, args.learning_rate, args.factor, args.max_grad_norm)

main()

