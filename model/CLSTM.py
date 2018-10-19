import torch
import torch.nn as nn

class Lstm(torch.nn.Module):
    def __init__(self, opt):
        super(Lstm, self).__init__()
        hidden_size = 256
        self.embedding = torch.nn.Embedding(opt.VOCAB_SIZE, 512)
        self.rnn = torch.nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 512, (2, 512)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((99, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 512, (3, 512)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((98, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 512, (4, 512)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((97, 1))
        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(3*hidden_size, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            nn.Dropout(0.5),
            torch.nn.Linear(512, opt.NUM_CLASS_1)
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(3*hidden_size, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            nn.Dropout(0.5),
            torch.nn.Linear(512, opt.NUM_CLASS_2)
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(3*hidden_size, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            nn.Dropout(0.5),
            torch.nn.Linear(512, opt.NUM_CLASS_3)
        )

    def forward(self, x):
        output = self.embedding(x)
        output = output.reshape([output.shape[0], 1, output.shape[1], output.shape[2]])
        output1 = self.conv1(output)
        output1 = output1.reshape([output1.shape[0], 1, output1.shape[1]])
        output2 = self.conv2(output)
        output2 = output2.reshape([output2.shape[0], 1, output2.shape[1]])
        output3 = self.conv3(output)
        output3 = output3.reshape([output3.shape[0], 1, output3.shape[1]])
        output = torch.cat((output1, output2, output3), dim=1)
        output, (h_n, c_n) = self.rnn(output)
        output = output.reshape([output.shape[0], output.shape[1] * output.shape[2]])
        
        output1 = self.fc1(output)
        output2 = self.fc2(output)
        output3 = self.fc3(output)
        return output1, output2, output3

