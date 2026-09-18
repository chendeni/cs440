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
import math
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
        h=170
        conv_window = 5
        conv_channels = 24
        conv_output_size = round((((math.sqrt(in_size/3)-conv_window+1)/3)**2)*conv_channels)
        #print(conv_output_size)
        self.h=h
        self.lrate = lrate
        self.in_size = in_size

        
        
            #nn.ReLU(),
        
        
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        self.conv_seq = nn.Sequential(
            nn.Conv2d(3,conv_channels,conv_window),
            nn.ReLU(),
            nn.MaxPool2d(3,3)
        )

        self.seq = nn.Sequential(
            nn.Linear(conv_output_size, h),
            nn.ReLU(),
            nn.Linear(h, out_size)
            
        )
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, momentum=0.9)


        
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        x_4d = x.reshape(len(x),3,31,31)
        a = self.conv_seq(x_4d)
        #print(len(a[0][0][0]))
        
        output = self.seq(torch.flatten(a,1))
        #print(output)
        #raise NotImplementedError("You need to write this part!")
        return output

        

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        
        if(not x.requires_grad):
            x.requires_grad = True
        self.optimizer.zero_grad()
        
        loss = self.loss_fn(x, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

        
'''
def image_reshape(image,width,channels):
    if(len(image)%(width*channels) != 0):
        print("image size not divisible by width or channels")
    length = len(image)/(width*channels)

    for i in range(channels):
        for j in range(width):
            for k in range(length):
'''



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

    std_train_set = (train_set - train_set.mean(dim=1, keepdim=True))/train_set.std(dim=1, keepdim=True)
    std_dev_set = (dev_set - dev_set.mean(dim=1, keepdim=True))/dev_set.std(dim=1, keepdim=True)

    train_dataset = get_dataset_from_arrays(std_train_set, train_labels)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False)
    
    
    net = NeuralNet(0.01, nn.CrossEntropyLoss(), len(train_set[0]), 4)
    losses=[]
    for i in range(epochs): #epochs
        total_loss=0
        
        for batch in train_loader:
            #print(len(batch['features']))
            #batch4d = batch['features'].reshape(len(batch['features']),3,31,31)
            
            
            outputs = net.forward(batch['features'])
            mean_loss = net.step(outputs,batch['labels'])
            total_loss += mean_loss
        losses.append(total_loss)

    #dev_set_4d = dev_set.reshape(len(dev_set),3,31,31)
    result = net.forward(dev_set).numpy(force=True)
    yhats = np.argmax(result,axis=1).astype(int)
    
    print(type(yhats[0]))
    return losses,yhats,net


