import torch
import torch.nn.functional as F
from torch.nn import Conv2d as Conv2d
from torch.nn import ConvTranspose2d as ConvTranspose2d
from torch.nn import MaxPool2d as MaxPool2d
import numpy as np


class MyUNet(torch.nn.Module):
    
    def __init__(self):
        super(MyUNet, self).__init__()
        self.u1_conv1 = Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, padding = 1)
        self.u1_conv2 = Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
        self.u1_pool  = MaxPool2d(kernel_size = 2)
        
        self.u2_conv1 = Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.u2_conv2 = Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
        self.u2_pool  = MaxPool2d(kernel_size = 2)
        
        self.u3_conv1 = Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.u3_conv2 = Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.u3_pool  = MaxPool2d(kernel_size = 2)
        
        self.u4_conv1 = Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.u4_conv2 = Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.u4_pool  = MaxPool2d(kernel_size = 2)
        
        self.u5_conv1 = Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.u5_conv2 = Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.u5_pool  = MaxPool2d(kernel_size = 2)
        
        self.u6_conv1 = Conv2d(in_channels  = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.u6_conv2 = Conv2d(in_channels  = 128, out_channels = 128, kernel_size = 3, padding = 1)
        
        self.u7_deconv1 = ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        #concat
        self.u7_conv1   = Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1)
        self.u7_conv2   = Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        
        self.u8_deconv1 = ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 2, stride = 2)
        #concat
        self.u8_conv1   = Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, padding = 1)
        self.u8_conv2   = Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        
        self.u9_deconv1 = ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 2)
        #concat
        self.u9_conv1   = Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1)
        self.u9_conv2   = Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        
        self.u10_deconv1 = ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 2, stride = 2)
        #concat
        self.u10_conv1   = Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 1)
        self.u10_conv2   = Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
        
        self.u11_deconv1 = ConvTranspose2d(in_channels = 16, out_channels = 8, kernel_size = 2, stride = 2)
        #concat
        self.u11_conv1   = Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, padding = 1)
        self.u11_conv2   = Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = 1)
        
        self.lastconv2d  = Conv2d(in_channels = 8, out_channels = 1, kernel_size= 1)
        
    
    def forward(self, x):
        c1 = F.relu(self.u1_conv2(F.relu(self.u1_conv1(x))))
        p1 = self.u1_pool(c1)
        
        c2 = F.relu(self.u2_conv2(F.relu(self.u2_conv1(p1))))
        p2 = self.u2_pool(c2)
        
        c3 = F.relu(self.u3_conv2(F.relu(self.u3_conv1(p2))))
        p3 = self.u3_pool(c3)
        
        c4 = F.relu(self.u4_conv2(F.relu(self.u4_conv1(p3))))
        p4 = self.u4_pool(c4)
        
        c5 = F.relu(self.u5_conv2(F.relu(self.u5_conv1(p4))))
        p5 = self.u5_pool(c5)
        
        c55 = F.relu(self.u6_conv2(F.relu(self.u6_conv1(p5))))
        
        u6 = self.u7_deconv1(c55)
        u6 = torch.cat((u6, c5), dim = 1)
        c6 = F.relu(self.u7_conv2(F.relu(self.u7_conv1(u6))))
        
        u71 = self.u8_deconv1(c6)
        u71 = torch.cat((u71, c4), dim = 1)
        c61 = F.relu(self.u8_conv2(F.relu(self.u8_conv1(u71))))
        
        u7 = self.u9_deconv1(c61)
        u7 = torch.cat((u7, c3), dim = 1)
        c7 = F.relu(self.u9_conv2(F.relu(self.u9_conv1(u7))))
        
        u8 = self.u10_deconv1(c7)
        u8 = torch.cat((u8, c2), dim = 1)
        c8 = F.relu(self.u10_conv2(F.relu(self.u10_conv1(u8))))
        
        u9 = self.u11_deconv1(c8)
        u9 = torch.cat((u9, c1), dim = 1)
        c9 = F.relu(self.u11_conv2(F.relu(self.u11_conv1(u9))))
        
        c10 = self.lastconv2d(c9)
        
        return torch.sigmoid(c10)
