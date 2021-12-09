import torch.optim as optim
import torch.nn.functional as F
import torch
from torch import nn
from Models.SimpleCNN import SimpleNet
from config import MOMENTUM, LEARNING_RATE, EPOCHS, LOG_INTERVAL
from loadData import getLoaders

import matplotlib.pyplot as plt

def train(dataName):

    trainLoader, testLoader = getLoaders(dataName)

    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM)

    for epoch in range(EPOCHS):
        ##################
        ### TRAIN LOOP ###
        ##################
        # set the model to train mode
        model.train()
        train_loss = 0
        for data, target in trainLoader:
            # clear the old gradients from optimized variables
            optimizer.zero_grad()
            # forward pass: feed inputs to the model to get outputs
            output = model(data)
            # calculate the training batch loss
            loss = criterion(output, target)
            # backward: perform gradient descent of the loss w.r. to the model params
            loss.backward()
            # update the model parameters by performing a single optimization step
            optimizer.step()
            # accumulate the training loss
            train_loss += loss.item()

        #######################
        ### VALIDATION LOOP ###
        #######################
        # set the model to eval mode
        model.eval()
        valid_loss = 0
        # turn off gradients for validation
        with torch.no_grad():
            for data, target in testLoader:
                # forward pass
                output = model(data)
                # validation batch loss
                loss = criterion(output, target)
                # accumulate the valid_loss
                valid_loss += loss.item()
        #########################
        ## PRINT EPOCH RESULTS ##
        #########################
        train_loss /= len(trainLoader)
        valid_loss /= len(testLoader)
        print(f'Epoch: {epoch + 1}/{EPOCHS}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}')

if __name__=="__main__":
    train("mnist")