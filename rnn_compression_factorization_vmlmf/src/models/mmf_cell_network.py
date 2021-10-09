
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

