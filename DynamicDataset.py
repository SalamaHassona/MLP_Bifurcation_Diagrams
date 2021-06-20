import torch
from torch.utils import data
import random
from DBcm import UseDatabase


class DynamicDataset(data.Dataset):

    def __init__(self, dataset_index, database_config, shuffle=False, batch_size=64):
        self.dataset_index = dataset_index
        self.database_config = database_config
        self.shuffle = shuffle
        # self.batch_size = batch_size

    def __len__(self):
        with UseDatabase(self.database_config['dbconfig']) as cursor:
            _SQL = """SELECT max({index}) FROM `timeseries`""".format(index=self.dataset_index)
            cursor.execute(_SQL)
            self.dataset_split_length = cursor.fetchall()[0][0] + 1
        return self.dataset_split_length
        # return 5

    def __getitem__(self, index):
        with UseDatabase(self.database_config['dbconfig']) as cursor:
            _SQL = """SELECT features, label FROM `timeseries` where {index}=%s""".format(index=self.dataset_index)
            cursor.execute(_SQL, (index,))
            self.batch = cursor.fetchall()

        if self.shuffle:
            random.shuffle(self.batch)

        features = list()
        labels = list()
        for item in self.batch:
            features.append([float(n) for n in item[0].split(',')])
            labels.append(item[1])
        X = self.standardization(features)
        y = torch.LongTensor(labels)
        return X, y

    def standardization(self, features):
        _X = torch.FloatTensor(features)
        _X_mean = _X.mean(dim=1).view(_X.shape[0], 1)
        _X_std = _X.std(dim=1).view(_X.shape[0], 1)
        X = (_X - _X_mean) / _X_std
        return X
