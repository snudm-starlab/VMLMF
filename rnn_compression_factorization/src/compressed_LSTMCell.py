import torch
import torch.nn as nn
class myLSTMGroupCell(nn.Module):
  
    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None,g=2,recurrent_init=None,
                 hidden_init=None,isShuffle=False):
        super(myLSTMGroupCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
        self.g=g
        self.isShuffle=isShuffle
        
        self.W_row=input_size if wRank is None else wRank
        self.W=None if wRank is None else nn.Parameter(0.1*torch.randn([self.input_size,self.wRank]))
        self.Ws=[]
        
        #case: only when uRank is not none
        #Ws=[W_f,W_i,W_c,W_o]
        for i in range(4): # of lstm cell gates
            self.Ws.append(nn.Parameter(0.1*torch.randn([self.W_row,hidden_size])))
        
        self.Ug=[[] for i in range(self.g)] # U, UU, UUU, UUUU ... > weight for group 1,2,3,4,...
        for idx,urank in enumerate(self.uRanks):
            self.Ug[idx].append(nn.Parameter(0.1*torch.randn([g,int(hidden_size/g),self.uRanks[idx]])))
            for i in range(4): #f,i,c,o
                self.Ug[idx].append(nn.Parameter(0.1*torch.randn([g,self.uRanks[idx],int(hidden_size/g)])))
        
        self.bias_f=nn.Parameter(torch.ones([1,hidden_size]))
        self.bias_i=nn.Parameter(torch.ones([1,hidden_size]))
        self.bias_c=nn.Parameter(torch.ones([1,hidden_size]))
        self.bias_o=nn.Parameter(torch.ones([1,hidden_size]))
            
    def hiddenOperation(self,g_h,groupWeight,batch_size):
        #g_h transposed and multiplied with U, groupWeight=gate's weight for each group
        g_h=torch.bmm(g_h,groupWeight)
        g_h=torch.transpose(g_h,0,1)
        g_h=g_h.contiguous().view(batch_size,self.hidden_size)
        return g_h
        
    def forgetgate(self,x,h):
        wVal_f=torch.matmul(x,self.Ws[0]) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[0])
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))
        
        uVal_f=0
        
        for g_idx,groupWeight in enumerate(self.Ug):
            #groupWeight=[U,U_f,U_i,U_c,U_o]
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,groupWeight[0])
            uVal_f+=self.hiddenOperation(g_h,groupWeight[1],batch_size) if self.uRanks[g_idx] > 0 else 0
        
        return torch.sigmoid(wVal_f+uVal_f+self.bias_f)
    
    def inputgate(self,x,h):
        wVal_i=torch.matmul(x,self.Ws[1]) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[1])
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))
        
        uVal_i=0
        
        for g_idx,groupWeight in enumerate(self.Ug):
            #groupWeight=[U,U_f,U_i,U_c,U_o]
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,groupWeight[0])
            uVal_i+=self.hiddenOperation(g_h,groupWeight[2],batch_size) if self.uRanks[g_idx] > 0 else 0
        
        return torch.sigmoid(wVal_i+uVal_i+self.bias_i)
        
    def outgate(self,x,h):
        wVal_o=torch.matmul(x,self.Ws[2]) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[2])
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))

        uVal_o=0

        for g_idx,groupWeight in enumerate(self.Ug):
            #groupWeight=[U,U_f,U_i,U_c,U_o]
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,groupWeight[0])
            uVal_o+=self.hiddenOperation(g_h,groupWeight[3],batch_size) if self.uRanks[g_idx] > 0 else 0

        return torch.sigmoid(wVal_o+uVal_o+self.bias_o)   
        
    def gate_gate(self,x,h):
        wVal_g=torch.matmul(x,self.Ws[3]) if self.wRank is None else torch.matmul(torch.matmul(x,self.W),self.Ws[3])
        batch_size=h.shape[0]
        gidx_list=list(range(self.g))
        hview=h.view(batch_size,self.g,int(self.hidden_size/self.g))
        
        uVal_g=0
        
        for g_idx,groupWeight in enumerate(self.Ug):
            #groupWeight=[U,U_f,U_i,U_c,U_o]
            index=gidx_list[g_idx:]+gidx_list[0:g_idx]
            g_h=hview[:,index,:]
            g_h=torch.transpose(g_h,0,1)
            g_h=torch.bmm(g_h,groupWeight[0])
            uVal_g+=self.hiddenOperation(g_h,groupWeight[4],batch_size) if self.uRanks[g_idx] > 0 else 0
        
        return torch.tanh(wVal_g+uVal_g+self.bias_c)   
        
    def forward(self,x,hiddenStates):
        (h,c)=hiddenStates
        '''
        print(self.Ws)
        for i in self.Ug:
            for j in i:
                print(j.shape)
        
        '''
        val_f=self.forgetgate(x,h)
        val_i=self.inputgate(x,h)
        val_o=self.outgate(x,h)
        val_c=self.gate_gate(x,h)
        
        c_next=val_f*c+val_i*val_c
        h_next=val_o*torch.tanh(c_next)
        
        c_next=self.shuffle(c_next) if self.isShuffle else c_next
        h_next=self.shuffle(h_next) if self.isShuffle else h_next
        
        return c_next,h_next
