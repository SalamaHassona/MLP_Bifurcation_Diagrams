from model import Network, train
from torch import optim
from torch import nn
from DynamicDataset import DynamicDataset
import torch

config = {'dbconfig': {'host': '127.0.0.1',
                       'user': 'root',
                       'password': 'root',
                       'database': 'timeseries_db',
                       'use_unicode': True}}

trainset = DynamicDataset("training_index_nn", config, shuffle=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
validset = DynamicDataset("validation_index_nn", config, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=False)
testset = DynamicDataset("test_index_nn", config, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

device = torch.device("cuda")
model = Network()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)
# state_dict = torch.load('checkpoint_epoch_10_trainloss_0.085_trainaccuracy_0.969_validloss_0.155_validaccuracy_0.961_epoch_12.pth')
# model.load_state_dict(state_dict)
try:
    train(model, trainloader, validloader, testloader, criterion, optimizer, device, epochs=10, start_epochs=0, save=True, save_file_name='checkpoint')
except Exception as ex:
    print(ex)
