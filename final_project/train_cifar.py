import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

import HamiltonianNet as hamNet


def train(dataset, architecture, path):
    # Hyper-parameters
    epoch_num = 200                   # Number of epochs
    batch_size = 128                  # Batch size
    weight_decay = 5e-4               # Weight decay factor (L2 regularization)
    weight_smoothness_decay = 2e-4    # Weight smoothness decay factor
    momentum = 0.9                    # Momentum
    initial_lr = 0.001                 # Initial learning rate
    lr_decay_factor = 0.1             # Learning rate decay factor
    lr_decay_epochs = [80, 120, 160]  # Epochs in which learning rate decays

    weight_cache = {}                 # Weight cache to store weights before optimizer step

    checkpoint_base_path = './checkpoints/hamiltonian_net_checkpoint-'

    # Get device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # Data transform
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == 'cifar10':
        train_val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'stl10':
        train_val_dataset = torchvision.datasets.STL10(root="./data", split='test', download=True, transform=transforms.ToTensor())
    else:
        raise NotImplementedError('Dataset {} is not supported!'.format(dataset))

    torch.manual_seed(43)
    val_size = int(len(train_val_dataset)*0.1)
    train_size = len(train_val_dataset) - val_size
    trainset, valset = random_split(train_val_dataset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size)

    if architecture == 'original_hamiltonian':
        net = hamNet.HamiltonianOriginalNetwork()
    elif architecture == 'inception_hamiltonian':
        net = hamNet.HamiltonianInceptionNetwork()
    else:
        raise NotImplementedError('Architecture {} is not supported!'.format(architecture))

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_epochs, gamma=lr_decay_factor)
    history = []
    for epoch in range(epoch_num):

        with tqdm(trainloader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                # Set data and labels to device, zero gradients
                inputs, labels = data.to(device), target.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs = net(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Add weight smoothness decay regularization
                regularization_term, weight_cache = weight_smoothness_reg(net, weight_cache)
                loss += weight_smoothness_decay * float(regularization_term)

                # Loss backward pass and optimizer step
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

            # Validation and saving checkpoint
            if epoch % 20 == 0:
                # Validation step
                with torch.no_grad():
                    result = evaluate(net, valloader, criterion)
                    net.train()
                print("\nEpoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
                history.append(result)

                # Save Checkpoint
                checkpoint_path = checkpoint_base_path + str(epoch) + '.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)

            # Advance scheduler
            scheduler.step()

    # Save network weights
    torch.save(net.state_dict(), path)

    # Plot accuracies and losses
    plot_accuracies(history)
    plot_losses(history)


def test(dataset, architecture, path):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'stl10':
        testset = torchvision.datasets.STL10(root="./data", split='test', download=True, transform=transforms.ToTensor())
    else:
        raise NotImplementedError('Dataset {} is not supported!'.format(dataset))

    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if architecture == 'original_hamiltonian':
        net = hamNet.HamiltonianOriginalNetwork()
    elif architecture == 'inception_hamiltonian':
        net = hamNet.HamiltonianInceptionNetwork()
    else:
        raise NotImplementedError('Architecture {} is not supported!'.format(architecture))

    # Set network to device and eval state, load weights
    net.to(device)
    net.load_state_dict(torch.load(path))
    net.eval()
    # Initialize statistics
    correct = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # Calculate outputs by running images through the network
            outputs = net(images)
            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# Utilities Functions
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
    acc = accuracy(outputs, labels)  # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(net, val_loader, criterion):
    net.eval()
    results = [validation_step(data, criterion, net) for data in val_loader]
    return validation_epoch_end(results)


def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.show()


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()


def weight_smoothness_reg(net, weight_cache):
    regularization_term = 0.0
    for name, param in net.named_parameters():
        if ('g_block' in name or 'f_block' in name) and 'weight' in name:
            weight_mat = param
            if name not in weight_cache.keys():
                weight_cache[name] = weight_mat
                continue
            weight_diff = torch.sub(weight_cache[name], weight_mat)
            norm = torch.norm(weight_diff, dim=(2, 3))
            regularization_term += sum(sum(norm))
            weight_cache[name] = torch.clone(weight_mat)
    return regularization_term, weight_cache


if __name__ == "__main__":
    train(dataset='cifar10', architecture='inception_hamiltonian', path='./hamiltonian_net2.pth')
    test(dataset='cifar10', architecture='inception_hamiltonian', path='./hamiltonian_net2.pth')
