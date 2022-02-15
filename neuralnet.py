# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques for the fall 2021 semester
# Modified by Kaiwen Hong for the Spring 2022 semester

"""
This is the main entry point for MP2. You should only modify code
within this file and neuralnet.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

from pickletools import optimize
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @param l(x,y) an () tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 2 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        super(NeuralNet, self).__init__()
        self.lrate = lrate        
        self.loss_fn = loss_fn
        self.net = torch.nn.Sequential(torch.nn.Linear(in_size, 32, bias = True), torch.nn.ReLU(), torch.nn.Linear(32, out_size, bias = True))
        
        # self.optims = torch.optim.SGD(self.parameters(), self.lrate)
        # raise NotImplementedError("You need to write this part!")

    # def set_parameters(self, params):
    #     """ Sets the parameters of your network.

    #     @param params: a list of tensors containing all parameters of the network
    #     """
    #     raise NotImplementedError("You need to write this part!")

    # def get_parameters(self):
    #     """ Gets the parameters of your network.

    #     @return params: a list of tensors containing all parameters of the network
    #     """
    #     raise NotImplementedError("You need to write this part!")

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # raise NotImplementedError("You need to write this part!")
        # return torch.ones(x.shape[0], 1)
        return self.net(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        optimizer = torch.optim.SGD(self.net.parameters(), self.lrate)
        find  = self.forward(x)

        loss_function = self.loss_fn(find, y)

        optimizer.zero_grad()
        loss_function.backward()
        optimizer.step()

        return loss_function.item()

        # raise NotImplementedError("You need to write this part!")
        # return 0.0


def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):
    """ Fit a neural net. Use the full batch size.

    @param train_set: an (N, out_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epoches: an int, the number of epoches of training
    @param batch_size: size of each batch to train on. (default 100)

    NOTE: This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    losses = []
    yhats = np.zeros(len(dev_set))

    # training
    # # find the lrate, in_size, and out_size
    nNet = NeuralNet(lrate = .035, loss_fn= torch.nn.CrossEntropyLoss(), in_size = len(train_set[0]), out_size=2)
    working_train_set = (train_set - train_set.mean()) / train_set.std()

    for i in range(n_iter) : #might need to go to n_iter - 1
        first = i * batch_size
        last = (i+1) * batch_size

        train = working_train_set[first : last]
        labels = train_labels[first : last]
        loss = nNet.step(train, labels)
        losses.append(loss)

    # # raise NotImplementedError("You need to write this part!")

    # development
    working_dev_set = (dev_set - train_set.mean()) / train_set.std()
    net = nNet(working_dev_set).detach().numpy()

    for i in range(len(net)) :
        yhats[i] = np.argmax(net[i])

    return losses, yhats, nNet
