import torch
from torch import nn
import torch.nn.functional as F
import time
# from torch.utils.tensorboard import SummaryWriter


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200001, 256)
        # self.fc1 = nn.Linear(200001, 2048)
        # self.fc2 = nn.Linear(2048, 1024)
        # self.fc3 = nn.Linear(1024, 512)
        # self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        # x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))
        # x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.dropout(F.relu(self.fc6(x)))
        x = F.log_softmax(self.fc7(x), dim=1)

        return x


def validation(model, loader, criterion, device):
    accuracy = 0
    loss = 0
    start = time.time()
    for features, labels in loader:
        features, labels = features.squeeze().to(device), labels.squeeze().to(device)
        output = model.forward(features)
        loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean().item()
    end = time.time()
    return loss, accuracy, end-start


def train(model, train_loader, validation_loader, test_loader, criterion, optimizer, device, epochs=5, start_epochs=0, save=False, save_file_name=None):
    # steps = 0
    # writer = SummaryWriter()

    for epoch in range(start_epochs, epochs):
        training_loss = 0
        training_accuracy = 0
        # Model in training mode, dropout is on
        start = time.time()
        model.train()
        for features, labels in train_loader:
            # steps += 1
            features, labels = features.squeeze().to(device), labels.squeeze().to(device)
            optimizer.zero_grad()

            output = model.forward(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            ps = torch.exp(output)
            equality = (labels == ps.max(1)[1])
            training_accuracy += equality.type_as(torch.FloatTensor()).mean().item()
        model.eval()
        end = time.time()
        print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
              "Training Loss: {:.3f} ".format(training_loss / len(train_loader)),
              "Training Accuracy: {:.3f} ".format(training_accuracy / len(train_loader)),
              "Training Time: {time}".format(time=str(end-start)))

        with torch.no_grad():
            valid_loss, valid_accuracy, valid_time = validation(model, validation_loader, criterion, device)
        print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
              "Validation Loss: {:.3f} ".format(valid_loss / len(validation_loader)),
              "Validation Accuracy: {:.3f}".format(valid_accuracy / len(validation_loader)),
              "Validation Time: {time}".format(time=str(valid_time)))

        # writer.add_graph(model)

        if save:
            torch.save(model.state_dict(),
                       '{name}_trainloss_{trainloss}_trainaccuracy_{trainaccuracy}_validloss_{validloss}_validaccuracy_{validaccuracy}_epoch_{epoch}.pth'
                       .format(name=save_file_name,
                               trainloss=str(round(training_loss / len(train_loader), 3)),
                               trainaccuracy=str(round(training_accuracy / len(train_loader), 3)),
                               validloss=str(round(valid_loss / len(validation_loader), 3)),
                               validaccuracy=str(round(valid_accuracy / len(validation_loader), 3)),
                               epoch=str(epoch)))
        model.train()
    model.eval()
    # writer.close()
    with torch.no_grad():
        test_loss, test_accuracy, test_time = validation(model, test_loader, criterion, device)
    print("Test Loss: {:.3f} ".format(test_loss / len(test_loader)),
          "Test Accuracy: {:.3f} ".format(test_accuracy / len(test_loader)),
          "Test Time: {time}".format(time=str(test_time)))