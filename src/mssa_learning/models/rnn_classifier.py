import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import preprocessing
import progressbar
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

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def _train_input_fn(self, features, label):
        hidden = self.initHidden()
        for i in range(features.size()[0]):
            output, hidden = self.forward(features[i], hidden)
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

                # forward + backward + optimize
                rand_input, rand_label = self.random_example(train_input, train_target)
                output = self._train_input_fn(rand_input, rand_label.unsqueeze(0))
                loss = self.criterion(output, rand_label)
                loss.backward()
                loss = loss.data[0]
                # self.optimizer.step()
                current_loss += loss

                # Add parameters' gradients to their values, multiplied by learning rate
                for p in self.parameters():
                    p.data.add_(-self.learning_rate, p.grad.data)

                # Print iter number, loss, name and guess
                if i % self.print_every == 0:
                    print('%d %d%% %.4f' % (iteration, iteration / float(self.steps * self.epochs) * 100.0, loss))

                # Add current loss avg to list of losses
                if i % self.plot_every == 0:
                    all_losses.append(current_loss / self.plot_every)
                    current_loss = 0
                iteration += 1
        return all_losses