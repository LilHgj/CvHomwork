from torch import nn
import torch
import torch.nn.functional as F

def build_model():
    model=FaceAtrr()
    loss = nn.CrossEntropyLoss()
    return model,loss

class FaceAtrr(nn.Module):
    def __init__(self):
        super(FaceAtrr, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                      in_channels=1,
                      out_channels=3,
                      kernel_size=(5, 5),
                      stride=1,
                      padding=2
                      ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2),
            )
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=1,
                      kernel_size=(5, 5),
                      stride=1,
                      padding=2
                      ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1=nn.Flatten() #32*32
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(32*32,512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4= nn.Linear(256, 42)
        # self.fc2=nn.Linear(,512)
        # self.fc2 = nn.Linear(512, 41)
        self.dropout=nn.Dropout(p=0.5)



    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.fc1(x)
        x=self.dropout(x)
        x=self.fc2(self.relu(x))
        x = self.dropout(x)
        x=self.fc3(self.relu(x))
        x = self.fc4(x)
        return x

