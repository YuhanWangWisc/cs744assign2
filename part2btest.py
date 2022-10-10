import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
from datetime import datetime as time
import torch.distributed as dist
import argparse

device = "cpu"
torch.set_num_threads(4)

batch_size = 256 # batch for one node
def train_model(model, train_loader, optimizer, criterion, epoch, args):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    running_loss = 0.0

    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):

        # zero the parameter gradients
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        for p in model.parameters():
            # synchronize gradients using allreduce
            torch.distributed.all_reduce(p.grad)
            p.grad = p.grad/args.size
        
        optimizer.step()

        running_loss += loss.item()
       
        if batch_idx % 20 == 19:    # print every 20 mini-batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            
def main():
    if (torch.distributed.is_available() == False):
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('--master-ip', dest='master_ip', type=str)
    parser.add_argument('--num-nodes', dest='size', type=int)
    parser.add_argument('--rank', dest='rank',type=int)
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="gloo", init_method=args.master_ip, world_size=args.size, rank=args.rank)
    print("successfully set up the process group")
    
    np.random.seed(0)
    torch.manual_seed(0)

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    sampler = torch.utils.data.distributed.DistributedSampler(training_set, num_replicas=args.size, rank=args.rank)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=int(batch_size/args.size),
                                                    sampler=sampler,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(15):
        if epoch == 1 :
            torch.save(model.state_dict(), 'torchmodel_weights_2b.pth')
        train_model(model, train_loader, optimizer, training_criterion, epoch, args)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    main()
