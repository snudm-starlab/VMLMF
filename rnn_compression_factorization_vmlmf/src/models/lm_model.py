import torch
import torch.nn as nn
import torch.nn.functional as F

#Embedding module.
class Embed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, embed_size))

    def forward(self, x):
        return self.W[x]

    def __repr__(self):
        return "Embedding(vocab: {}, embedding: {})".format(self.vocab_size, self.embed_size)
class myHMD(nn.Module):
    def __init__(self, input_size, hidden_size,rich_rank,sparse_rank, dropout = 0, winit = 0.1,device=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.A_x = nn.Parameter(torch.Tensor(4 * rich_rank, input_size))
        self.B_x = nn.Parameter(torch.Tensor(4 * (hidden_size-rich_rank), sparse_rank))
        self.C_x = nn.Parameter(torch.Tensor(4 * rich_rank, input_size))
       
        self.A_h = nn.Parameter(torch.Tensor(4 * rich_rank, hidden_size))
        self.B_h = nn.Parameter(torch.Tensor(4 * (hidden_size-rich_rank), sparse_rank))
        self.C_h = nn.Parameter(torch.Tensor(4 * rich_rank, hidden_size))
       
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))
       
        
    
    def __repr__(self):
        return "LSTM(input: {}, hidden: {})".format(self.input_size, self.hidden_size)

    def lstm_step(self, x, h, c):
        #print("device of the model:".format(self.A_x.device))
       
        fa,ia,ga,oa=torch.matmul(self.A_x,x.T).chunk(4,0) #[32,6],[81,6]
        fb,ib,gb,ob=self.B_x.chunk(4,0)
        fc,ic,gc,oc=torch.matmul(self.C_x,x.T).chunk(4,0) #[12,6],[81,6]
       
        fa_h,ia_h,ga_h,oa_h=torch.matmul(self.A_h,h.T).chunk(4,0) #[32,6],[81,6]
        fb_h,ib_h,gb_h,ob_h=self.B_h.chunk(4,0)
        fc_h,ic_h,gc_h,oc_h=torch.matmul(self.C_h,h.T).chunk(4,0) #[12,6],[81,6]
       
        fb_x,ib_x,gb_x,ob_x=self.b_x.chuck(4,0)
        fb_h,ib_h,gb_h,ob_h=self.b_h.chunk(4,0)
       
        forgetgate=torch.sigmoid(fb_x+bf_h+torch.cat(fa,torch.matmul(fb,fc))+torch.cat(fa_h,torch.matmul(fb_h,fc_h)))
        inputgate=torch.sigmoid(ib_x+ib_h+torch.cat(ia,torch.matmul(ib,ic))+torch.cat(ia_h,torch.matmul(ib_h,ic_h)))
        newgate=torch.tanh(gb_x+gb_h+torch.cat(ga,torch.matmul(gb,gc))+torch.cat(ga_h,torch.matmul(gb_h,gc_h)))
        outputgate=torch.sigmoid(ob_x+ob_h+torch.cat(oa,torch.matmul(ob,oc))+torch.cat(oa_h,torch.matmul(ob_h,oc_h)))
       
        c = forgetgate * c + inputgate * newgate
        h = outputgate * torch.tanh(c)
        return h, c

    #Takes input tensor x with dimensions: [T, B, X].
    def forward(self, x, states):
        h, c = states
        #print(">>>>>>>> x.shape:",x.shape)
        #print(">>>>>>>> h shape:",h.shape)
        outputs = []
        inputs = x.unbind(0)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c)
            outputs.append(h)
        return torch.stack(outputs), (h, c)


######## <End of custom vmlmf lstm model code > ############

class myVMLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0, winit = 0.1,wRank=None,uRanks=None,device=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.wRank=wRank
        self.uRanks=uRanks

        #U in LMF
        self.U_x=nn.Parameter(torch.Tensor(input_size,wRank))
        self.U_h=nn.Parameter(torch.Tensor(hidden_size,uRanks))

        #V in LMF
        self.W_x = nn.Parameter(torch.Tensor(4 * hidden_size, wRank))
        self.W_h = nn.Parameter(torch.Tensor(4 * hidden_size, uRanks))

        #bias
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))

        #diagonal vector
        self.dia_x=nn.Parameter(torch.Tensor(1,input_size))
        self.dia_h=nn.Parameter(torch.Tensor(1,hidden_size))

        #save navigator
        self.cnt=0


    def __repr__(self):
        return "LSTM(input: {}, hidden: {})".format(self.input_size, self.hidden_size)

    def lstm_step(self, x, h, c, W_x, W_h, b_x, b_h):
        dev = next(self.parameters()).device
        #print("device",dev )
        """
        x and h.shape: [20,200] (batch, #hid)
        vm_refined:[20,800] (batch,4gates) vm_x,h:[20,800] lowered_x,h: [20,800]
        """
        #save vm - redundant values
        vm_refined_x=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)
        vm_refined_h=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)
        """if self.cnt==0:
            print(">>>>>>> device:{}/{}".format(vm_refined_x.device,vm_refined_h.device))"""
        #vm (for all 4 gates)
        vm_x=self.dia_x*x.squeeze()
        vm_h=self.dia_h*h.squeeze()
        vm_x=torch.cat([vm_x for _ in range(4)],dim=1)
        vm_h=torch.cat([vm_h for _ in range(4)],dim=1)
        """if self.cnt==0:
            print(">>>>>>> shape:{}/{}".format(vm_x.shape,vm_h.shape))  """
        #lmf(x*U -> x'*V) 
        lowered_x=torch.matmul(torch.matmul(x,self.U_x),self.W_x.t())
        lowered_h=torch.matmul(torch.matmul(h,self.U_h),self.W_h.t())
        """if self.cnt==0:
            print(">>>>>>> shape:{}/{}".format(lowered_x.shape,lowered_h.shape))"""
        #cal redundant values
        hidden_size=self.hidden_size
        input_size=self.input_size
        for gate_idx in range(0,4*hidden_size,hidden_size):
            vm_refined_x[:,gate_idx:gate_idx+input_size]=x*torch.sum((self.U_x*self.W_x[gate_idx:gate_idx+input_size,:]),dim=1)
            vm_refined_h[:,gate_idx:gate_idx+hidden_size]=h*torch.sum((self.U_h*self.W_h[gate_idx:gate_idx+hidden_size,:]),dim=1)
        gx=vm_x+lowered_x-vm_refined_x+self.b_x
        gh=vm_h+lowered_h-vm_refined_h+self.b_h
        """if self.cnt==0:
            print(">>>>>>> shape:{}/{}".format(gx.shape,gh.shape))"""
            

        #total
        #gx = torch.addmm(b_x, x, W_x.t())
        #gh = torch.addmm(b_h, h, W_h.t())
        xi, xf, xo, xn = gx.chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)
        """if self.cnt==0:
            print(">>>>>>> shape:{}/{}/{}/{}\n>>>>>>> shape:{}/{}/{}/{}".format(xi.shape,xf.shape,xo.shape,xn.shape,hi.shape,hf.shape,ho.shape,hn.shape))
            self.cnt+=1"""
        inputgate = torch.sigmoid(xi + hi)
        forgetgate = torch.sigmoid(xf + hf)
        outputgate = torch.sigmoid(xo + ho)
        newgate = torch.tanh(xn + hn)
        c = forgetgate * c + inputgate * newgate
        h = outputgate * torch.tanh(c)
        return h, c

    #Takes input tensor x with dimensions: [T, B, X].
    def forward(self, x, states):
        h, c = states
        outputs = []
        inputs = x.unbind(0)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.W_x, self.W_h, self.b_x, self.b_h)
            outputs.append(h)
        return torch.stack(outputs), (h, c)
#My custom written LSTM module.
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0, winit = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.W_x = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.W_h = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))

    def __repr__(self):
        return "LSTM(input: {}, hidden: {})".format(self.input_size, self.hidden_size)

    def lstm_step(self, x, h, c, W_x, W_h, b_x, b_h):
        gx = torch.addmm(b_x, x, W_x.t())
        gh = torch.addmm(b_h, h, W_h.t())
        xi, xf, xo, xn = gx.chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)
        inputgate = torch.sigmoid(xi + hi)
        forgetgate = torch.sigmoid(xf + hf)
        outputgate = torch.sigmoid(xo + ho)
        newgate = torch.tanh(xn + hn)
        c = forgetgate * c + inputgate * newgate
        h = outputgate * torch.tanh(c)
        return h, c

    #Takes input tensor x with dimensions: [T, B, X].
    def forward(self, x, states):
        h, c = states
        #print(">>>>>>>> x.shape:",x.shape)
        #print(">>>>>>>> h shape:",h.shape)
        outputs = []
        inputs = x.unbind(0)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.W_x, self.W_h, self.b_x, self.b_h)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

class Linear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x):
        #.view() flattens the input which has dimensionality [T,B,X] to dimenstionality [T*B, X].
        z = torch.addmm(self.b, x.view(-1, x.size(2)), self.W.t())
        return z

    def __repr__(self):
        return "FC(input: {}, output: {})".format(self.input_size, self.hidden_size)


class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, layer_num, dropout, winit, wRank=None,uRanks=None,lstm_type = "pytorch",cell=myVMLSTMCell_NEO_faster,device=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.winit = winit
        self.lstm_type = lstm_type
        self.embed = Embed(vocab_size, hidden_size)
        
        
        if lstm_type =="hmd":
            self.rnns=[myHMD(hidden_size, hidden_size,rich_rank=wRank,sparse_rank=uRanks,device=device) for i in range(layer_num)]
            self.rnns = nn.ModuleList(self.rnns)
        elif lstm_type!="vmlmf":
            self.rnns = [LSTM(hidden_size, hidden_size) if lstm_type == "custom" else nn.LSTM(hidden_size, hidden_size) for i in range(layer_num)]
            self.rnns = nn.ModuleList(self.rnns)
        else:
            self.rnns = [myVMLSTM(hidden_size, hidden_size,wRank=wRank,uRanks=uRanks,device=device) for i in range(layer_num)]
            self.rnns = nn.ModuleList(self.rnns)

        self.fc = Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()
        
    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.winit, self.winit)
            
    def state_init(self, batch_size):
        dev = next(self.parameters()).device
        states = [(torch.zeros(batch_size, layer.hidden_size, device = dev), torch.zeros(batch_size, layer.hidden_size, device = dev)) if self.lstm_type == "custom" or self.lstm_type =="vmlmf" or self.lstm_type == "hmd"
                  else (torch.zeros(1, batch_size, layer.hidden_size, device = dev), torch.zeros(1, batch_size, layer.hidden_size, device = dev)) for layer in self.rnns]
        return states
    
    def detach(self, states):
        return [(h.detach(), c.detach()) for (h,c) in states]
    
    def forward(self, x, states):
        x = self.embed(x)
        x = self.dropout(x)
        for i, rnn in enumerate(self.rnns):
            #print("states[i]shape",states[i][0].shape)
            x, states[i] = rnn(x, states[i])
            x = self.dropout(x)
        scores = self.fc(x)
        return scores, states
