
class myMMFCell_g2(nn.Module):

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, g=2, recurrent_init=None,hidden_init=None):
        super(myMMFCell_g2, self).__init__()
        print("MMF with no g_structure for input2hid - last updated 10.08")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRanks = wRank
        self.uRanks = uRanks
        self.g = g

        self.layers=nn.ParameterDict()
        
        # vm_vector
        self.layers['dia_x']=nn.Parameter(0.1 * torch.randn([1,input_size]))
        self.layers['dia_h']=nn.Parameter(0.1 * torch.randn([1,hidden_size]))
        
        # UV of input to hid matrix in LMF
        self.layers['Ux']=nn.Parameter(0.1 * torch.randn([input_size,wRank]))
        self.layers['Vx'] = nn.Parameter(0.1 * torch.randn([4 * hidden_size, wRank]))

        # UV of hid to hid matrix in LMF
        for g_idx in range(self.g):
            self.layers['Uh_{}'.format(g_idx)]=nn.Parameter(0.1*torch.randn([g,int(hidden_size/g),uRanks[g_idx]])) # weight sharing across 4 gates
            self.layers['Vh_{}'.format(g_idx)]=nn.Parameter(0.1*torch.randn([g,uRanks[g_idx],4*int(hidden_size/g)])) # V matrix of each gate  

        for vec in ['x','h']:
            self.layers['bias_{}'.format(vec)]= nn.Parameter(torch.ones([1, 4*hidden_size]))
    
    def __repr__(self):
        return "LSTM VM Group (input:{}, hidden:{}, wRank:{}, uRanks:{})".format(self.input_size,self.hidden_size,self.wRank,self.uRanks)

    def forward(self, x, hiddenStates,device):

        dev=next(self.parameters()).device
        (h, c) = hiddenStates
        batch_size=h.shape[0]

        #VM operation
        vm_x=torch.cat([self.layers['dia_x']*x.squeeze(),torch.zeros([h.shape[0],self.hidden_size-self.input_size],device=dev)],dim=1) if self.hidden_size>=self.input_size else None
        vm_h=self.layers['dia_h']*h.squeeze()

        #LMF_x operation 
        lowered_x=torch.matmul(torch.matmul(x,self.layers['Ux']),self.layers['Vx'].t())
        ## x and h.shape[0] == batch_size 81 
        vm_refined_x=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)
        vm_refined_h=torch.zeros(h.shape[0],4*self.hidden_size,device=dev)
        #print(f"Uh shape:{self.layers['Uh_0'].shape}")
        vm_refined_Uh=self.layers['Uh_0'].view(self.hidden_size,self.uRanks[0])
        #print(f">>> Vh_0 shape[g,urank,4*h/g] <-> {self.layers['Vh_0'].shape}")
        vm_refined_Vh=torch.transpose(self.layers['Vh_0'],1,2).contiguous()
        #print(f">>> Vh_0 shape[g,urank,4*h/g] not influenced<-> {self.layers['Vh_0'].shape}")
        #print(f">>> vm_refined_Vh shape[g,4*h/g,urank] <-> {vm_refined_Vh.shape}")

        #vm_refined_Vh=vm_refined_Vh.view(-1,self.uRanks[0])
        #print(f">>> vm_refined_Vh shape[g,4*h/g,urank] <-> {vm_refined_Vh.shape}")
        for gate_idx in range(0,4*self.hidden_size,self.hidden_size): 
            vm_refined_x[:,gate_idx:gate_idx+self.input_size]=x*torch.sum((self.layers['Ux']*self.layers['Vx'][gate_idx:gate_idx+self.input_size,:]),dim=1)
            #vm_refined_h[:,gate_idx:gate_idx+self.hidden_size]=h*torch.sum((vm_refined_Uh*vm_refined_Vh[gate_idx:gate_idx+self.hidden_size,:]),dim=1)
            gate_g_idx,gate_g_size=int(gate_idx/self.g),int(self.hidden_size/self.g)
            gate_Vh=vm_refined_Vh[:,gate_g_idx:gate_g_idx+gate_g_size,:].view(-1,self.uRanks[0])
            vm_refined_h[:,gate_idx:gate_idx+self.hidden_size]=h*torch.sum((vm_refined_Uh*gate_Vh),dim=1) #10.04

        gx=lowered_x-vm_refined_x+self.layers['bias_x']
        xi, xf, xo, xn = gx.chunk(4, 1)
        

        #LMF_h operation
        ## 1. LMF for each group
        ## 2. compute redundant values from g0
        
        index = list(range(self.g))
        # h for each group operation
        
        for i in range(self.g):
            h_op = h.view(batch_size, self.g, int(self.hidden_size / self.g))  
            index=index[1:]+index[0:1] if i>0 else index
            h_op=h_op[:,index,:] if i > 0 else h_op
            h_op=torch.transpose(h_op,0,1) #[g,batch_size,h/g]

            Uh=self.layers['Uh_{}'.format(i)] #[g,h/g,uRanks]
            h_op=torch.bmm(h_op,Uh) #[g,batch_size,uRanks]
            Vh=self.layers['Vh_{}'.format(i)] #[g,uRanks,h/g*4]
            h_op=torch.bmm(h_op,Vh) #[g,batch_size,h/g*4]
            h_op=torch.transpose(h_op,0,1) #[batch_size,g,h/g*4]
            h_op_chunked=h_op if i==0 else h_op_chunked+h_op
            
        f_h,i_h,n_h,o_h=h_op_chunked.chunk(4,dim=2)
        f_h=f_h.contiguous().view(batch_size,self.hidden_size)
        i_h=i_h.contiguous().view(batch_size,self.hidden_size)
        n_h=n_h.contiguous().view(batch_size,self.hidden_size)
        o_h=o_h.contiguous().view(batch_size,self.hidden_size)

        gh=self.layers['bias_h']-vm_refined_h 
        hf,hi,hn,ho=gh.chunk(4,1)

        hf=hf+f_h; hi=hi+i_h; hn=hn+n_h; ho=ho+o_h

        inputgate = torch.sigmoid(xi + hi+vm_x+vm_h)
        forgetgate = torch.sigmoid(xf + hf+vm_x+vm_h)
        outputgate = torch.sigmoid(xo + ho+vm_x+vm_h)
        newgate = torch.tanh(xn +hn+ vm_x + vm_h)
        c_next = forgetgate * c + inputgate * newgate
        h_next = outputgate * torch.tanh(c_next)
        return h_next, c_next


class myMMFgCell_g2(nn.Module):

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, g=2, recurrent_init=None,hidden_init=None):
        super(myMMFgCell_g2, self).__init__()
        print("MMF without cmpl_vector - last updated 10.08")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRanks = uRanks
        self.g = g

        self.layers=nn.ParameterDict()
        
        # UV of input to hid matrix in LMF
        self.layers['Ux']=nn.Parameter(0.1 * torch.randn([input_size,wRank]))
        self.layers['Vx'] = nn.Parameter(0.1 * torch.randn([4 * hidden_size, wRank]))

        # UV of hid to hid matrix in LMF
        for g_idx in range(self.g):
            self.layers['Uh_{}'.format(g_idx)]=nn.Parameter(0.1*torch.randn([g,int(hidden_size/g),uRanks[g_idx]])) # weight sharing across 4 gates
            self.layers['Vh_{}'.format(g_idx)]=nn.Parameter(0.1*torch.randn([g,uRanks[g_idx],4*int(hidden_size/g)])) # V matrix of each gate  

        for vec in ['x','h']:
            self.layers['bias_{}'.format(vec)]= nn.Parameter(torch.ones([1, 4*hidden_size]))
    
    def __repr__(self):
        return "LSTM VM Group (input:{}, hidden:{}, wRank:{}, uRanks:{})".format(self.input_size,self.hidden_size,self.wRank,self.uRanks)

    def forward(self, x, hiddenStates,device):

        dev=next(self.parameters()).device
        (h, c) = hiddenStates
        batch_size=h.shape[0]

        #LMF_x operation 
        lowered_x=torch.matmul(torch.matmul(x,self.layers['Ux']),self.layers['Vx'].t())
        gx=lowered_x+self.layers['bias_x']
        xf, xi, xn, xo = gx.chunk(4, 1)
        

        #LMF_h operation
        ## 1. LMF for each group
        ## 2. compute redundant values from g0
        
        index = list(range(self.g))
        # h for each group operation
        
        for i in range(self.g):
            h_op = h.view(batch_size, self.g, int(self.hidden_size / self.g))  
            index=index[1:]+index[0:1] if i>0 else index
            h_op=h_op[:,index,:] if i > 0 else h_op
            h_op=torch.transpose(h_op,0,1) #[g,batch_size,h/g]

            Uh=self.layers['Uh_{}'.format(i)] #[g,h/g,uRanks]
            h_op=torch.bmm(h_op,Uh) #[g,batch_size,uRanks]
            Vh=self.layers['Vh_{}'.format(i)] #[g,uRanks,h/g*4]
            h_op=torch.bmm(h_op,Vh) #[g,batch_size,h/g*4]
            h_op=torch.transpose(h_op,0,1) #[batch_size,g,h/g*4]
            h_op_chunked=h_op if i==0 else h_op_chunked+h_op
            
        f_h,i_h,n_h,o_h=h_op_chunked.chunk(4,dim=2)
        f_h=f_h.contiguous().view(batch_size,self.hidden_size)
        i_h=i_h.contiguous().view(batch_size,self.hidden_size)
        n_h=n_h.contiguous().view(batch_size,self.hidden_size)
        o_h=o_h.contiguous().view(batch_size,self.hidden_size)

        gh=self.layers['bias_h']
        hf,hi,hn,ho=gh.chunk(4,1)
        hf=hf+f_h; hi=hi+i_h; hn=hn+n_h; ho=ho+o_h

        inputgate = torch.sigmoid(xi + hi)
        forgetgate = torch.sigmoid(xf + hf)
        outputgate = torch.sigmoid(xo + ho)
        newgate = torch.tanh(xn +hn)
        c_next = forgetgate * c + inputgate * newgate
        h_next = outputgate * torch.tanh(c_next)
        return h_next, c_next

class myMMFcCell_g2(nn.Module):

    def __init__(self, input_size, hidden_size, wRank=None, uRanks=None, g=2, recurrent_init=None,hidden_init=None):
        super(myMMFcCell_g2, self).__init__()
        print("MMF without g_structure - last updated 10.08_")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        self.wRank = wRank
        self.uRank= uRanks
        print("u",uRanks)
        self.layers=nn.ParameterDict()
        
        for vec in ['x','h']:
            sz,rank=(input_size,wRank) if vec =='x' else (hidden_size,uRanks)
            self.layers[f'dia_{vec}']= nn.Parameter(0.1 * torch.randn([1,sz]))
            self.layers[f'U{vec}']= nn.Parameter(0.1 * torch.randn([sz,rank]))
            self.layers[f'V{vec}'] = nn.Parameter(0.1 * torch.randn([4 * hidden_size, rank]))
            self.layers[f'bias_{vec}']= nn.Parameter(torch.ones([1, 4*hidden_size]))
    
    def __repr__(self):
        return "LSTM VM Group (input:{}, hidden:{}, wRank:{}, uRanks:{})".format(self.input_size,self.hidden_size,self.wRank,self.uRank)

    def forward(self, x, hiddenStates,device):

        dev=next(self.parameters()).device
        (h, c) = hiddenStates
        batch_size=h.shape[0]

        #VM operation
        vm_x=torch.cat([self.layers['dia_x']*x.squeeze(),torch.zeros([h.shape[0],self.hidden_size-self.input_size],device=dev)],dim=1) if self.hidden_size>=self.input_size else None
        vm_h=self.layers['dia_h']*h.squeeze()

        #LMF_x operation _ output [batch,4*hidden_size]
        lowered_x=torch.matmul(torch.matmul(x,self.layers['Ux']),self.layers['Vx'].t()) 
        lowered_h=torch.matmul(torch.matmul(h,self.layers['Uh']),self.layers['Vh'].t()) 

        ## x and h.shape[0] == batch_size 81 
        vm_refined_x=torch.zeros(x.shape[0],4*self.hidden_size,device=dev)
        vm_refined_h=torch.zeros(h.shape[0],4*self.hidden_size,device=dev)

        for gate_idx in range(0,4*self.hidden_size,self.hidden_size): 
            vm_refined_x[:,gate_idx:gate_idx+self.input_size]=x*torch.sum((self.layers['Ux']*self.layers['Vx'][gate_idx:gate_idx+self.input_size,:]),dim=1)
            vm_refined_h[:,gate_idx:gate_idx+self.hidden_size]=h*torch.sum((self.layers['Uh']*self.layers['Vh'][gate_idx:gate_idx+self.hidden_size,:]),dim=1)

        gx=lowered_x-vm_refined_x+self.layers['bias_x']
        gh=lowered_h-vm_refined_h+self.layers['bias_h']

        xi, xf, xo, xn = gx.chunk(4, 1)        
        hi, hf, ho, hn=gh.chunk(4,1)

        inputgate = torch.sigmoid(xi + hi+vm_x+vm_h)
        forgetgate = torch.sigmoid(xf + hf+vm_x+vm_h)
        outputgate = torch.sigmoid(xo + ho+vm_x+vm_h)
        newgate = torch.tanh(xn +hn+ vm_x + vm_h)

        c_next = forgetgate * c + inputgate * newgate
        h_next = outputgate * torch.tanh(c_next)
        return h_next, c_next

