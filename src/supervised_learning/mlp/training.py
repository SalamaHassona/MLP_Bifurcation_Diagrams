import argparse
import mlp_neural_net_1
import mlp_neural_net_3
import mlp_neural_net_6
from torch import optim
from torch import nn
from src.supervised_learning.utils.welding_arc_system.DynamicDataset import DynamicDataset
from src.supervised_learning.utils.welding_arc_system.generate_training_data_RLC import DataGenerator
import torch
import time
from torch.utils.tensorboard import SummaryWriter


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
    return loss, accuracy, end - start


def train(model, train_loader, validation_loader, test_loader, criterion, optimizer, device, epochs=5,
          start_epochs=0, save=False, save_file_name=None):
    # steps = 0
    writer = SummaryWriter()

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

            output = model.forward(features, device)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            ps = torch.exp(output)
            equality = (labels == ps.max(1)[1])
            training_accuracy += equality.type_as(torch.FloatTensor()).mean().item()
        model.eval()
        end = time.time()
        writer.add_scalar("Loss: train", training_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy: train", training_accuracy / len(train_loader), epoch)
        print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
              "Training Loss: {:.3f} ".format(training_loss / len(train_loader)),
              "Training Accuracy: {:.3f} ".format(training_accuracy / len(train_loader)),
              "Training Time: {time}".format(time=str(end - start)))

        with torch.no_grad():
            valid_loss, valid_accuracy, valid_time = validation(model, validation_loader, criterion, device)
        writer.add_scalar("Loss: validation", valid_loss / len(validation_loader), epoch)
        writer.add_scalar("Accuracy: validation", valid_accuracy / len(validation_loader), epoch)
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
    writer.add_scalar("Loss: test", test_loss / len(test_loader))
    writer.add_scalar("Accuracy: test", test_accuracy / len(test_loader))
    writer.flush()
    writer.close()
    print("Test Loss: {:.3f} ".format(test_loss / len(test_loader)),
          "Test Accuracy: {:.3f} ".format(test_accuracy / len(test_loader)),
          "Test Time: {time}".format(time=str(test_time)))


def main():
    generator = DataGenerator(labeled_data_file=args.labeled_data_file, data_util_file=args.data_util_file,
                              threshold=args.threshold, dt=args.dt, L=args.L, tmin=args.tmin, tmax=args.tmax)
    training_data, validation_data, test_data = generator.get_data(batch_size=args.batch_size,
                                                                   ts_nth_element=args.ts_nth_element,
                                                                   training_frac=0.7,
                                                                   validation_frac=0.15)
    trainset = DynamicDataset(training_data, shuffle=True, batch_size=args.batch_size)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
    validset = DynamicDataset(validation_data, shuffle=True, batch_size=args.batch_size)
    validloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=False)
    testset = DynamicDataset(test_data, shuffle=True, batch_size=args.batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    device = torch.device("cuda:0")
    if args.mlp_model == "mlp_1":
        model = mlp_neural_net_1.Network()
    elif args.mlp_model == "mlp_3":
        model = mlp_neural_net_3.Network()
    else:
        model = mlp_neural_net_6.Network()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)
    try:
        train(model, trainloader, validloader, testloader, criterion, optimizer, device,
              epochs=args.epochs, start_epochs=args.start_epochs,
              save=True, save_file_name=args.save_file_name)
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--tmin", type=int, default=1800)
    p.add_argument("--tmax", type=int, default=2000)
    p.add_argument("--ts_nth_element", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--threshold", type=int, default=23)
    p.add_argument("--mlp_model", type=str, default="mlp_1")
    p.add_argument("--labeled_data_file", type=str, default="labeled_data_file.txt")
    p.add_argument("--data_util_file", type=str, default="data_util_file.txt")
    p.add_argument("--save_file_name", type=str, default="checkpoint")
    p.add_argument("--start_epochs", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end - start))
