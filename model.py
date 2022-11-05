#%%

import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%

class new_resnet50(nn.Module):
    def __init__(self):
        super(new_resnet50, self).__init__()

        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(2048, 512)

    def forward(self, x):

        out = self.model(x)
        
        return out

class projectionnet(nn.Module):
    def __init__(self):
        super(projectionnet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

    def forward(self, x):

        out = self.net(x)
        
        return out    

class downstreamnet(nn.Module):
    def __init__(self):
        super(downstreamnet, self).__init__()

        self.net = torch.nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)
        )

    def forward(self, x):

        out = self.net(x)
        
        return out     