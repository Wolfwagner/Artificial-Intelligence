# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
     def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.conv1 = nn.Sequential(nn.Conv2d(3, 31,  3),nn.ReLU(),nn.MaxPool2d(2, 2),
                                   nn.Conv2d(31, 64, 3),nn.ReLU(),nn.MaxPool2d(2, 2))
        self.input = nn.Linear(64*6*6, 64)
        self.output = nn.Linear(64, out_size)
        self.optims = optim.Adam(self.parameters(), lr=lrate, weight_decay=0.0)    

     def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """        

        x = x.view(-1, 3, 31, 31)   
        x = self.conv1(x)
        z = x.size(0)
        x = x.view(z, -1) 
        x = self.input(x)
        y = self.output(x)
        return y

     def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        
        self.optims.zero_grad()
        out= self.forward(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optims.step()
        
        return loss.item()
    



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    losses= []
    yhats = []
    
    loss_function = nn.CrossEntropyLoss()
    net = NeuralNet(0.001, loss_function, len(train_set[0]), 4)
    train_mean= torch.mean(train_set)
    train_std = torch.std(train_set)
    dev_mean= torch.mean(dev_set)
    dev_std = torch.std(dev_set)
    
    train_standard = (train_set-train_mean)/train_std
    dev_standard = (dev_set-dev_mean)/dev_std

    for epoch in range(epochs): 
        data = train_standard[(batch_size*epoch)%train_set.shape[0]:batch_size*(epoch+1)]
        labels = train_labels[(batch_size*epoch)%train_set.shape[0]:batch_size*(epoch+1)]

        loss_val = net.step(data, labels)
        losses.append(loss_val)
    
    for i in dev_standard:
        y = torch.argmax(net.forward(i))
        yhats.append(y.item())
    yhats = np.array(yhats)
    assert(len(yhats) == dev_set.shape[0])
    
    return losses, yhats, net
