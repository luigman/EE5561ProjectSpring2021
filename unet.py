import logging
from typing import List
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#Debug tip from: https://discuss.pytorch.org/t/how-do-i-print-output-of-each-layer-in-sequential/5773/3
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.channels = [1, 64, 128, 256, 512, 1024]

        self.encoder1 = nn.Sequential(
            # PrintLayer(),
            #Layer 1 - input [64, 1, 572, 572]
            nn.Conv2d(self.channels[0], self.channels[1], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[1], self.channels[1], 3, stride=1, padding=0),
            nn.ReLU(inplace=True))
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2,stride=2),
            # PrintLayer(),
            #Layer 2 - input [64, 64, 284, 284]
            nn.Conv2d(self.channels[1], self.channels[2], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[2], self.channels[2], 3, stride=1, padding=0),
            nn.ReLU(inplace=True))
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2,stride=2),
            # PrintLayer(),
            #Layer 3 - input [64, 129, 140, 140]
            nn.Conv2d(self.channels[2], self.channels[3], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[3], self.channels[3], 3, stride=1, padding=0),
            nn.ReLU(inplace=True))
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2,stride=2),
            # PrintLayer(),
            #Layer 4 - input [64, 256, 68, 68]
            nn.Conv2d(self.channels[3], self.channels[4], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[4], self.channels[4], 3, stride=1, padding=0),
            nn.ReLU(inplace=True))
        self.encoder5 = nn.Sequential(
            nn.MaxPool2d(2,stride=2),
            # PrintLayer(),
            #Layer 5 - input [64, 512, 32, 32]
            nn.Conv2d(self.channels[4], self.channels[5], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[5], self.channels[5], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # PrintLayer(),
            #smallest size - input [64, 1024, 28, 28]

            nn.Upsample(size=(55,55), mode='bilinear', align_corners=False),
            #H_in = H_out-kernel_size+1 so H_in = 56-2+1 = 55
            torch.nn.ConvTranspose2d(self.channels[5], self.channels[4], 2, stride=1, padding=0, output_padding=0)
        )

        self.decoder4 = nn.Sequential(
            # PrintLayer(),
            #Layer 4 - input [64, 1024, 56, 56]
            nn.Conv2d(self.channels[5], self.channels[4], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[4], self.channels[4], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(103,103), mode='bilinear', align_corners=False),
            #H_in = H_out-kernel_size+1 so H_in = 103
            torch.nn.ConvTranspose2d(self.channels[4], self.channels[3], 2, stride=1, padding=0, output_padding=0)
        )
        self.decoder3 = nn.Sequential(
            # PrintLayer(),
            #Layer 3 - input [64, 512, 104, 104]
            nn.Conv2d(self.channels[4], self.channels[3], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[3], self.channels[3], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(199,199), mode='bilinear', align_corners=False), 
            #H_in = H_out-kernel_size+1 so H_in = 199
            torch.nn.ConvTranspose2d(self.channels[3], self.channels[2], 2, stride=1, padding=0, output_padding=0)
        )
        self.decoder2 = nn.Sequential(
            # PrintLayer(),
            #Layer 2 - input [64, 256, 200, 200]
            nn.Conv2d(self.channels[3], self.channels[2], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[2], self.channels[2], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(391,391), mode='bilinear', align_corners=False),
            torch.nn.ConvTranspose2d(self.channels[2], self.channels[1], 2, stride=1, padding=0, output_padding=0)
        )
        self.decoder1 = nn.Sequential(
            # PrintLayer(),
            #Layer 3 - input [64, 128, 392, 392]
            nn.Conv2d(self.channels[2], self.channels[1], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[1], self.channels[1], 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[1], 2, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # self.crop1 = nn.Sequential(transforms.CenterCrop(392))
        # self.crop2 = nn.Sequential(transforms.CenterCrop(200))
        # self.crop3 = nn.Sequential(transforms.CenterCrop(104))
        # self.crop4 = nn.Sequential(transforms.CenterCrop(56))

    def centerCrop(self, x, size):
        w, h = x.shape[-2:]
        assert w == h
        start = (w-size)//2
        end = (w+size)//2
        assert end-start == size
        return x[:,:,start:end,start:end]

    def forward(self, x):
        encoder1 = self.encoder1(x.float())
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)

        encoder1_crop = self.centerCrop(encoder1,392)
        encoder2_crop = self.centerCrop(encoder2,200)
        encoder3_crop = self.centerCrop(encoder3,104)
        encoder4_crop = self.centerCrop(encoder4,56)

        decoder5 = torch.cat((encoder4_crop, encoder5), dim=1)
        decoder4 = self.decoder4(decoder5)
        decoder4 = torch.cat((encoder3_crop, decoder4), dim=1)
        decoder3 = self.decoder3(decoder4)
        decoder3 = torch.cat((encoder2_crop, decoder3), dim=1)
        decoder2 = self.decoder2(decoder3)
        decoder2 = torch.cat((encoder1_crop, decoder2), dim=1)
        decoder1 = self.decoder1(decoder2)

        return decoder1
