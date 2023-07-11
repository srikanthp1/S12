# Custom Resnet Model with prep layer, residual blocks with convolutions, 4x4 Max pooling and FC layer
import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom_resNet(nn.Module):
    
    def __init__(self):
        super(Custom_resNet, self).__init__()
        self.in_planes = 64
        
        #Prep Layer
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        #Layer1
        
        ##Layer1-Part A
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()   
        )
        
        ##Layer1-Part B
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        #Layer2
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2), 
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        ##Layer3-Part A
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        ##Layer3-Part B
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        ## MaxPooling of 4x4
        self.max = nn.Sequential(
            nn.MaxPool2d(4, 4)
        )
        
        # Fully Connected Layer
        self.linear = nn.Linear(512*1, 10)

    def forward(self, x):
        
        out = self.prep(x)

        X = self.conv2(out)
        R1 = self.conv3(X)
        out = X + R1
        
        out = self.conv4(out)

        X2 = self.conv5(out)
        R2 = self.conv6(X2)
        out = X2 + R2
        
        out = self.max(out)
        
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)

        return out