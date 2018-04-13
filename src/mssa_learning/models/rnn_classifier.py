import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import preprocessing
import random


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, steps=100000, epochs=2, scale=False):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
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

    def forward(self, input):
        combined = torch.cat((input, self.hidden), 1)
        self.hidden = self.i2h(combined)
        y = self.i2o(combined)
        output = self.softmax(y)
        return output

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def _train_input_fn(self, features):
        for i in range(features.size()[0]):
            output = self.forward(features[i])
        return output

    def random_example(self, train_input, train_target):
        r = random.randint(0, len(train_target) - 1)
        data = train_input[r]
        x = torch.zeros(len(data), 1, len(data[0]))
        for row in range(len(data)):
            x[row, 0] = torch.from_numpy(data[row]).float()
        x = Variable(x)
        y = Variable(torch.from_numpy(np.array([train_target[r]])))
        return x, y

    def predict(self, input_data):
        """
        Predict the label given input variable using a forward pass and get the
        largest index
        """
        x = torch.zeros(len(input_data), 1, len(input_data[0]))
        for row in range(len(input_data)):
            x[row, 0] = torch.from_numpy(input_data[row]).float()
        x = Variable(x)
        for i in range(x.size()[0]):
            output = self.forward(x[i])
        top_n, top_i = output.data.topk(1)
        y_pred = int(top_i[0][0])
        return y_pred

    def fit(self, train_input, train_target):
        current_loss = 0
        all_losses = []

        iteration = 0
        for e in range(self.epochs):  # loop over the dataset multiple times
            for i in range(self.steps):
                self.hidden = self.init_hidden()
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                rand_input, rand_label = self.random_example(train_input, train_target)
                output = self._train_input_fn(rand_input)
                loss = self.criterion(output, rand_label)
                loss.backward()
                loss = loss.data[0]
                self.optimizer.step()
                current_loss += loss

                # Add parameters' gradients to their values, multiplied by learning rate
                # for p in self.parameters():
                #     p.data.add_(-self.learning_rate, p.grad.data)

                # Print iter number, loss, name and guess
                if i % self.print_every == 0:
                    print('%d %d%% %.4f' % (iteration, iteration / float(self.steps * self.epochs) * 100.0, loss))

                # Add current loss avg to list of losses
                if i % self.plot_every == 0:
                    all_losses.append(current_loss / self.plot_every)
                    current_loss = 0
                iteration += 1
        return all_losses
