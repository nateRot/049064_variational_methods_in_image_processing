import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

import HamiltonianNet as hamNet


def train():
    # Hyper-parameters
    epoch_num = 200
    batch_size = 100
    weight_decay = 2e-4
    weight_smoothness_decay = 2e-4
    momentum = 0.9
    initial_lr = 0.01
    lr_decay_factor = 0.1
    lr_decay_epochs = [80, 120, 160]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())

    torch.manual_seed(43)
    val_size = int(len(dataset)*0.1)
    train_size = len(dataset) - val_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size)

    net = hamNet.HamiltonianOriginalNetwork()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_epochs, gamma=lr_decay_factor)
    history = []
    for epoch in range(epoch_num):
        with tqdm(trainloader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, labels = data.to(device), target.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                #l1_reg = torch.tensor(0., requires_grad=True)
                #for name, param in net.named_parameters():
                #    if 'weight' in name:
                #        l1_reg = l1_reg + torch.norm(param, 1)
                #loss = loss + weight_smoothness_decay * l1_reg
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            if epoch % 20 == 0:
                # Validation step
                with torch.no_grad():
                    result = evaluate(net, valloader, criterion)
                print("\nEpoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
                history.append(result)

            # Advance scheduler
            scheduler.step()
    # Save network weights
    path = './hamiltonian_net.pth'
    torch.save(net.state_dict(), path)
    plot_accuracies(history)
    plot_losses(history)


def test():
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path = './hamiltonian_net.pth'
    net = hamNet.HamiltonianOriginalNetwork()
    net.to(device)
    net.load_state_dict(torch.load(path))
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def validation_epoch_end(results):
    batch_losses = [x['val_loss'] for x in results]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x['val_acc'] for x in results]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


def validation_step(data,  criterion, net):
    weight_smoothness_decay = 2e-4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)  # Generate predictions
    loss = criterion(outputs, labels)  # Calculate loss
    #l1_reg = torch.tensor(0., requires_grad=True)
    #for name, param in net.named_parameters():
    #    if 'weight' in name:
    #        l1_reg = l1_reg + torch.norm(param, 1)
    #loss = loss + weight_smoothness_decay * l1_reg
    acc = accuracy(outputs, labels)  # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(net, val_loader, criterion):
    results = [validation_step(data, criterion, net) for data in val_loader]
    return validation_epoch_end(results)


def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');




if __name__ == "__main__":
    train()
    test()
