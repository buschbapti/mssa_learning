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
    def __init__(self, input_size, hidden_size, output_size, batch_size=1, steps=100000, epochs=2, scale=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.use_gpu = False

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden2label = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.learning_rate = 0.005
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.steps = steps
        self.epochs = epochs
        self.print_every = 5000
        self.plot_every = 1000

        self.scaler = preprocessing.MinMaxScaler()
        self.scale = scale

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        return (h0, c0)

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        output = self.softmax(y)
        return output

    def random_example(self, train_input, train_target):
        r = random.randint(0, len(train_target) - 1)
        data = train_input[r]
        x = torch.zeros(len(data), self.batch_size, len(data[0]))
        for row in range(len(data)):
            x[row, 0] = torch.from_numpy(data[row]).float()
        x = Variable(x)
        y = Variable(torch.from_numpy(np.array([train_target[r]])))
        return x, y

    def predict(self, inputs):
        """
        Predict the label given input variable using a forward pass and get the
        largest index
        """
        top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
        category_i = top_i[0][0]
        return int(category_i)

    def fit(self, train_input, train_target):
        current_loss = 0
        all_losses = []

        iteration = 0
        for e in range(self.epochs):  # loop over the dataset multiple times
            for i in range(self.steps):
                # zero the parameter gradients
                self.optimizer.zero_grad()
                self.hidden = self.init_hidden()

                # forward + backward + optimize
                rand_input, rand_label = self.random_example(train_input, train_target)
                output = self.forward(rand_input)
                loss = self.criterion(output, rand_label)
                loss.backward()
                loss = loss.data[0]
                self.optimizer.step()
                current_loss += loss

                # Print iter number, loss, name and guess
                if i % self.print_every == 0:
                    print('%d %d%% %.4f' % (iteration, iteration / float(self.steps * self.epochs) * 100.0, loss))

                # Add current loss avg to list of losses
                if i % self.plot_every == 0:
                    all_losses.append(current_loss / self.plot_every)
                    current_loss = 0
                iteration += 1
        return all_losses