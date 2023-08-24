import preprocessing
from network import Network
import torch
import torch.nn as nn
import torch.optim as optim
from constants import *


def main():
    train, val, test = preprocessing.preprocess_data()

    network = Network()
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()  # Binary cross entropy loss

    for epoch in range(EPOCHS):
        for X, y in train:
            network.zero_grad()
            output = network.forward(X.view(-1, FEATURES))
            loss = criterion(output, y.view(-1, 1))
            loss.backward()
            optimizer.step()

        print(loss)  # Why is the loss not decreasing? Is the NN doing any learning?


if __name__ == '__main__':
    main()
