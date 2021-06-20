import torch
from torch.utils import data
import random


class DynamicDataset(data.Dataset):

    def standardization(self, features):
        _X=torch.FloatTensor(features)
        _X_mean = _X.mean(dim=1).view(_X.shape[0],1)
        _X_std = _X.std(dim=1).view(_X.shape[0],1)
        X=(_X-_X_mean)/_X_std
        return X

    def __init__(self, xdata, shuffle=False, batch_size=64):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.xdata = xdata

    def __len__(self):
        return self.xdata.shape[0] // self.batch_size
        # return 5

    def __getitem__(self, index):
        self.batch = self.xdata[self.batch_size*index: self.batch_size*(index+1), :]

        if self.shuffle:
            random.shuffle(self.batch)

        X = self.standardization(torch.from_numpy(self.batch[:,1:]).type(torch.FloatTensor))
        y = torch.from_numpy(self.batch[:,0]).type(torch.LongTensor)
        return X, y