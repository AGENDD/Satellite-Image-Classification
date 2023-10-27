import numpy
import torch
from torch import nn
import torch.optim as optim

class Modell(nn.Module):


    def __init__(self):
        super(Modell, self).__init__()
        self.name = "Model-2"
        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(

            nn.Linear(128 * 6 * 6, 128), 
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, 4),
            nn.Softmax(dim=1)
        )


    
    def forward(self, x):

        x = self.features(x)

        x = x.view(x.shape[0], -1)

        output = self.classifier(x)

        
        return output
