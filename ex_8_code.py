from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

"""

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


#for epoch in range(1, 10 + 1):
train(epoch=1)
test()

"""


class FirstNet(nn.Module):
    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(image_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        # self.drop = nn.Dropout2d()
        self.bn = nn.BatchNorm1d(image_size)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = x.view(-1, self.image_size)

        x = self.bn(x)

        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)

        x= self.bn2(x)
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(args, epoch, model, optimizer, train_loader, train_idx):
    model.train()
    sumLoss = 0
    correct = 0
    for batch_idx, (data, labals) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labals)
        sumLoss += F.nll_loss(output, labals, size_average=False).item()

        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labals.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #            100. * len(train_loader.dataset) / len(train_loader.dataset), loss.item()))
    sumLoss /= len(train_idx)
    # sumLoss /= 48000
    print('\ntrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        sumLoss, correct, len(train_idx),
        100. * correct / len(train_idx)))
    # print ('loss of train',sumLoss)
    return sumLoss


def testOnValid(args, model, validation_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labals in validation_loader:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += F.nll_loss(output, labals, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(labals.view_as(pred)).sum().item()

        valid_loss /= len(validation_loader.dataset)
    print('\nvalidaition set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(validation_loader),
        100. * correct / len(validation_loader)))
    return valid_loss


def test(args, model, test_loader, numOfParam, what, output):
    f = open(output, "w")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            f.write(str(pred.item()) + "\n")
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= numOfParam
        # test_loss /= len(test_loader.dataset)
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            what, test_loss, correct, len(test_loader),
            100. * correct / len(test_loader)))
        # print('len',len(test_loader.dataset))
        return test_loss

    # model.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in test_loader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
    #         pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #
    # test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

    # test_loss /= 12000
    # print('\nvalidaition set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(validation_loader),
    #     100. * correct / len(validation_loader)))
    # return  test_loss


# def testOnValid(args,model,validation_Loader,validation_idx):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in validation_Loader:
#         output = model(data)
#         test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#         test_loss /= len(validation_Loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#     test_loss, correct, len(validation_Loader.dataset), 100. * correct / len(validation_Loader.dataset)))
#     return test_loss/validation_idx


def main():
    # ## Define our MNIST Datasets (Images and Labels) for training and testing
    train_dataset = datasets.FashionMNIST(root='./data',
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

    test_dataset = datasets.FashionMNIST(root='./data',
                                         train=False,
                                         transform=transforms.ToTensor())

    ## We need to further split our training dataset into training and validation sets.

    # Define the indices
    indices = list(range(len(train_dataset)))  # start with all the indices in training set
    split = (int)(0.2 * (len(indices)))  # define the split size

    # Define your batch_size
    batch_size = 24

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Create the train_loader -- use your real batch_size which you
    # I hope have defined somewhere above
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, sampler=train_sampler)

    # You can use your above batch_size or just set it to 1 here.  Your validation
    # operations shouldn't be computationally intensive or require batching.
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=1, sampler=validation_sampler)

    # You can use your above batch_size or just set it to 1 here.  Your test set
    # operations shouldn't be computationally intensive or require batching.  We
    # also turn off shuffling, although that shouldn't affect your test set operations
    # either
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)
    # print(len(train_loader.dataset))

    # Training settings
    for data, target in test_loader:
        # data, target = data.to(device), target.to(device)
        #output = model(data)
        #test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
          # get the index of the max log-probability
        print(target.view_as)
        #correct += pred.eq(target.view_as(pred)).sum().item()
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    print(args)

    # train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
    #                                                           transform=transforms), batch_size=64, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms),
    #                                           batch_size=64, shuffle=True)

     use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

     device = torch.device("cuda" if use_cuda else "cpu")

     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = FirstNet(image_size=28 * 28)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    validationPerEphoc = []
    trainLoss = []
    testLoss = []
    for epoch in range(1, args.epochs + 1):
        print("epoch number: ", epoch)
        trainLoss.append(train(args, epoch, model, optimizer, train_loader, train_idx))
        validationPerEphoc.append(test(args, model, validation_loader, 12000, "validation", "validation.pred"))
    testLoss.append(test(args, model, test_loader, 10000, "test", "test.pred"))

    print(validationPerEphoc)

    print(trainLoss)
    # test(args, model, test_loader)

    plt.title(" Avg Loss Per Epoch")
    line1, = plt.plot(trainLoss, "green", label='train')
    line2, = plt.plot(validationPerEphoc, "orange", label='validation')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Avg loss")
    plt.show()


if __name__ == '__main__':
    main()

"""
def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
"""