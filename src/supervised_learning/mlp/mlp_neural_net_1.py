from torch import nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2500, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
