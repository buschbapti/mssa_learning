#!/usr/bin/env python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from mssa_learning.models.classifier import Classifier


class LSTMClassifier(Classifier):
    """Class representing an LSTM based classifier"""
    def __init__(self, input_size, hidden_size, output_size, seq_len, num_layers=1, batch_size=200, chunk_size=10, epochs=2, scale=False, save_folder=None):
        super(LSTMClassifier, self).__init__(input_size, hidden_size, output_size, "lstm", num_layers, batch_size, epochs, scale, save_folder)
        self.chunk_size = chunk_size
        self.use_gpu = False
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.hidden2label = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.learning_rate = 0.005
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def _init_hidden(self, size=None):
        """
        Initialize the hidden layer of the LSTM network

        Keywoard arguments:
        size -- Optional, batch size of the input data, if None set to the default batch size value

        Return arguments:
        (h0, c0) -- Tuple representing the hidden layer of the network
        """
        if size is None:
            size = self.batch_size
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.num_layers, size, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(self.num_layers, size, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, size, self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers, size, self.hidden_size))
        return (h0, c0)

    def _forward(self, input):
        """
        Apply the forward method of the neural network to the input data

        Keywoard arguments:
        input -- The single vector of features in entry of the network

        Return arguments:
        output -- Value returned by the last layer of the network
        """

        input = torch.transpose(input, 0, 1)
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        output = self.softmax(y)
        return output

    def predict(self, inputs):
        """
        Predict the label given input variable using a forward pass and get the
        largest index

        Keywoard arguments:
        inputs -- The set of features to predict

        Return:
        predicted -- The set of predicted categories
        """
        # initialize hiddent state
        self.hidden = self._init_hidden(len(inputs))
        inputs = Variable(inputs)
        # predict
        output = self._forward(inputs)
        # extract prediction by taking the max of the predicted vector
        predicted = []
        for i in range(len(inputs)):
            top_n, top_i = output.data[i].topk(1) # Tensor out of Variable with .data
            predicted.append(int(top_i[0]))
        return predicted

    def fit(self, train_set, test_set=None):
        """
        Fit the network to the train set in input

        Keywoard arguments:
        train_set -- Tuple containing the features to train on
        and their corresponding labels
        test_set -- Optional, tuple containing the features to evaluate
        the learned model on and their corresponding labels
        """
        current_loss = 0
        self.cumulative_loss = []
        self.cumulative_eval = []
        iteration = 0
        try:
            for e in range(self.epochs):  # loop over the dataset multiple times
                for batch_idx, (data, target) in enumerate(train_set):
                    data, target = Variable(data), Variable(target)
                    # zero the parameter gradients
                    self.hidden = self._init_hidden()
                    # truncated backprogation
                    input_parts = torch.split(data, self.chunk_size, dim=1)
                    for input_part in input_parts:
                        self.optimizer.zero_grad()
                        self.hidden[0].detach_()
                        self.hidden[1].detach_()
                        output = self._forward(input_part)
                        target = target.view(target.numel())
                        loss = self.criterion(output, target)
                        loss.backward()
                        loss = loss.data[0]
                        self.optimizer.step()
                        current_loss += loss
                    if iteration % self.print_every == 0:
                        self._score(iteration, current_loss, test_set)
                        current_loss = 0
                    iteration += 1

        except KeyboardInterrupt:
            print("Learning process interrupted")
        finally:
            # save results to file
            self._save_results()
            self._save_model()
