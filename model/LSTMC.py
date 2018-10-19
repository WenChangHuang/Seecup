import torch
import torch.nn as nn

class Lstm(torch.nn.Module):
    def __init__(self, opt):
        super(Lstm, self).__init__()
        hidden_size = 1024
        n_chanel = 256
        self.embedding = torch.nn.Embedding(opt.VOCAB_SIZE, 1024)
        self.rnn = torch.nn.LSTM(
            input_size=`1024,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_chanel, (2, hidden_size)),
            nn.BatchNorm2d(n_chanel),
            nn.ReLU(True),
            nn.MaxPool2d((99, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_chanel, (3, hidden_size)),
            nn.BatchNorm2d(n_chanel),
            nn.ReLU(True),
            nn.MaxPool2d((98, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, n_chanel, (4, hidden_size)),
            nn.BatchNorm2d(n_chanel),
            nn.ReLU(True),
            nn.MaxPool2d((97, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, n_chanel, (1, hidden_size)),
            nn.BatchNorm2d(n_chanel),
            nn.ReLU(True),
            nn.MaxPool2d((100, 1))
        )


        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(4 * n_chanel, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            nn.Dropout(0.5),
            torch.nn.Linear(512, opt.NUM_CLASS_1)
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(4 * n_chanel, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            nn.Dropout(0.5),
            torch.nn.Linear(512, opt.NUM_CLASS_2)
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(4 * n_chanel, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            nn.Dropout(0.5),
            torch.nn.Linear(512, opt.NUM_CLASS_3)
        )

    def forward(self, x):
        emb = self.embedding(x)
        output, (h_n, c_n) = self.rnn(emb)
        output = output.reshape([output.shape[0], 1, output.shape[1], output.shape[2]])
        output1 = self.conv1(output)
        output2 = self.conv2(output)
        output3 = self.conv3(output)
        output4 = self.conv4(output)
        output = torch.cat((output1, output2, output3, output4), dim=1)
        output = output.view(output.size()[0], -1)
        output1 = self.fc1(output)
        output2 = self.fc2(output)
        output3 = self.fc3(output)
        return output1, output2, output3
