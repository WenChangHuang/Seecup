import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class TextCNN(nn.Module):
    def __init__(self, opt):
        super(TextCNN, self).__init__()
        self.encoder = nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, (3, opt.EMBEDDING_DIM, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((98, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 128, (4, opt.EMBEDDING_DIM, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((97, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 128, (2, opt.EMBEDDING_DIM, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((99, 1))
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(3 * 128, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASS_1)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(3 * 128, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, opt.NUM_CLASS_2)
        )
        self.fc_3 = nn.Sequential(
            nn.Linear(3 * 128, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, opt.NUM_CLASS_3)
        )

    def forward(self, x):
        outputs = self.encoder(x)
        outputs = outputs.reshape([outputs.shape[0], 1, outputs.shape[1], outputs.shape[2]])
        output1 = self.conv1(outputs)
        
        output2 = self.conv2(outputs)

        output3 = self.conv3(outputs)
        
        outputs = torch.cat((output1, output2, output3), dim=1)
        outputs = outputs.view(outputs.size()[0], -1)
        output_1 = self.fc_1(outputs)
        output_2 = self.fc_2(outputs)
        output_3 = self.fc_3(outputs)
        return output_1, output_2, output_3

