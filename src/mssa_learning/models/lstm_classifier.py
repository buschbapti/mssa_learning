#!/usr/bin/env python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import preprocessing
import progressbar
import random
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size, num_layers=1, batch_size=20, chunk_size=10, steps=100000, epochs=2, scale=False):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_gpu = False
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.hidden2label = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.learning_rate = 0.005
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.steps = steps
        self.epochs = epochs
        self.print_every = 5000
        self.plot_every = 1000


        self.scalers = []
        for d in range(input_size):
            self.scalers.append(preprocessing.MinMaxScaler())
        self.scale = scale

    def init_hidden(self, size=None):
        if size is None:
            size = self.batch_size
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.num_layers, size, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(self.num_layers, size, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, size, self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers, size, self.hidden_size))
        return (h0, c0)

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        output = self.softmax(y)
        return output

    def random_examples(self, train_input, train_target):
        x = torch.zeros(self.seq_len, self.batch_size, self.input_size)
        y = []
        for i in range(self.batch_size):
            r = random.randint(0, len(train_target) - 1)
            data = train_input[r]
            for row in range(len(data)):
                x[row, i] = torch.from_numpy(data[row]).float()
            y.append(train_target[r])
        x = Variable(x)
        y = Variable(torch.from_numpy(np.array(y)))
        return x, y

    def predict(self, inputs):
        """
        Predict the label given input variable using a forward pass and get the
        largest index
        """
        # first normalize the data
        if self.scale:
            inputs = self.normalize_data(inputs)
        # initialize hiddent state
        self.hidden = self.init_hidden(len(inputs))
        # convert input to torch variables
        x = torch.zeros(self.seq_len, len(inputs), self.input_size)
        for i in range(len(inputs)):
            data = inputs[i]
            for row in range(len(data)):
                x[row, i] = torch.from_numpy(data[row]).float()
        x = Variable(x)
        # predict
        output = self.forward(x)
        # extract prediction by taking the max of the predicted vector
        predicted = []
        for i in range(len(inputs)):
            top_n, top_i = output.data[i].topk(1) # Tensor out of Variable with .data
            predicted.append(int(top_i[0]))
        return predicted

    def normalize_data(self, input_data, train=False):
        scaled_input = np.zeros((len(input_data), self.seq_len, self.input_size))
        for d in range(self.input_size):
            values = input_data[:, :, d]
            if train:
                self.scalers[d] = self.scalers[d].fit(values)
            scaled_input[:, :, d] = self.scalers[d].transform(values)
        return scaled_input

    def evaluate(self, inputs, targets):
        predicted = self.predict(inputs)
        nb_errors = 0
        nb_elem = len(inputs)
        for i, p in enumerate(predicted):
            if p != targets[i]:
                nb_errors += 1
        success_rate = (nb_elem - nb_errors) / float(nb_elem)
        return success_rate


    def fit(self, train_input, train_target, test_set=None):
        current_loss = 0
        all_losses = []
        if self.scale:
            train_input = self.normalize_data(train_input, train=True)
        iteration = 0
        for e in range(self.epochs):  # loop over the dataset multiple times
            for i in range(self.steps):
                # zero the parameter gradients
                self.hidden = self.init_hidden()
                # forward + backward + optimize
                rand_input, rand_target = self.random_examples(train_input, train_target)
                # truncated backprogation
                input_parts = torch.split(rand_input, self.chunk_size, dim=0)
                for input_part in input_parts:
                    self.optimizer.zero_grad()
                    self.hidden[0].detach_()
                    self.hidden[1].detach_()
                    output = self.forward(input_part)
                    loss = self.criterion(output, rand_target)
                    loss.backward()
                    loss = loss.data[0]
                    self.optimizer.step()
                    current_loss += loss
                # Print iter number, loss, name and guess
                if i % self.print_every == 0:
                    print('%d %d%% %.4f' % (iteration, iteration / float(self.steps * self.epochs) * 100.0, loss))
                    if test_set is not None:
                        print('success rate: %f' % (self.evaluate(test_set[0], test_set[1])))
                # Add current loss avg to list of losses
                if i % self.plot_every == 0:
                    all_losses.append(current_loss / self.plot_every)
                    current_loss = 0

                iteration += 1
        return all_losses