
# Code for implementing DeepConvLSTM
class DeepConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first=True, recurrent_inits=None,
                 hidden_inits=None, wRank=None, uRank=None, **kwargs):
        super(DeepConvLSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 1))
        self.conv2 = nn.Conv2d(64, 64, (5, 1))
        self.conv3 = nn.Conv2d(64, 64, (5, 1))
        self.conv4 = nn.Conv2d(64, 64, (5, 1))

        # self.lstm1 = nn.LSTM(7232, 128, batch_first = True)
        # self.lstm2 = nn.LSTM(128, 128, batch_first = True)
        self.lstm = myLSTM(7232, hidden_layer_sizes=[128, 128], batch_first=True)
        self.gru = myGRU(7232, hidden_layer_sizes=[128, 128], batch_first=True)
        # self.gru1 = nn.LSTM(7232, 128)
        # self.gru2 = nn.LSTM(128, 128)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, hidden=None):
        self.device = x.device
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))

        x, h = self.gru(x)

        """
        h0 = torch.zeros(1, x.size(0), 128).to(self.device)
        c0 = torch.zeros(1, x.size(0), 128).to(self.device)

        #print(x.shape)
        output, (h, c) = self.lstm1(x, (h0, c0))
        #print(output.shape)

        h1 = torch.zeros(1, output.size(0), 128).to(self.device)
        c1 = torch.zeros(1, output.size(0), 128).to(self.device)

        output, (h, c) = self.lstm2(output, (h1, c1))
        #output = output.permute(1,0,2)
        #output = output[0,:,:]
        """
        #########################################
        return x, h

# Code for implementing DeepConvLSTM
# This is implementation of DeepcConvolutional part, and LSTM part will be added
class DeepConv(nn.Module):
    def __init__(self, filter_size=5, filter_count=64):
        super(DeepConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 1))
        self.conv2 = nn.Conv2d(64, 64, (5, 1))
        self.conv3 = nn.Conv2d(64, 64, (5, 1))
        self.conv4 = nn.Conv2d(64, 64, (5, 1))

        # self.lstm1 = nn.LSTM(7232, 128, batch_first = True)
        # self.lstm2 = nn.LSTM(128, 128, batch_first = True)
        # self.lstm = myLSTM(7232, hidden_layer_sizes=[128, 128], batch_first = True)
        # self.gru = myGRU(7232, hidden_layer_sizes=[128, 128], batch_first = True)
        # self.gru1 = nn.LSTM(7232, 128)
        # self.gru2 = nn.LSTM(128, 128)

    def forward(self, x, hidden=None):
        self.device = x.device
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))

        return x
