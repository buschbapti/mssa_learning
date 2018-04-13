import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import preprocessing


class DNNClassifier(torch.nn.Module):
    def __init__(self, input_length, nb_classes, H=10, batch=100, steps=12000, epochs=2, scale=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(DNNClassifier, self).__init__()

        self.batch_size = batch  # size of each batch of data
        self.steps = steps
        self.epochs = epochs

        self.input = nn.Linear(input_length, H)
        self.hidden = nn.Linear(H, H)
        self.output = nn.Linear(H, nb_classes)

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)
        self.scaler = preprocessing.MinMaxScaler()
        self.scale = scale

        self.print_every = 2000
        self.plot_every = 1000

    def _train_input_fn(self, features, labels):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        x = features
        y = np.array(labels).reshape(len(labels),1)
        dataset = np.hstack((x, y))

        # Shuffle, repeat, and batch the examples.
        np.random.shuffle(dataset)
        dataset = dataset[:self.batch_size]

        # extract x and y from the dataset
        x = Variable(torch.from_numpy(dataset[:, :-1]).float())
        y = Variable(torch.from_numpy(dataset[:, -1]).long())

        # Return the output
        return x, y

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h1 = F.relu(self.input(x))
        h2 = F.relu(self.hidden(h1))
        x = self.output(h2)
        return F.log_softmax(x, dim=0)

    def predict(self, inputs):
        """
        Predict the label given input variable using a forward pass and get the
        largest index
        """
        if self.scale:
            input_data = self.scaler.transform(inputs)
        else:
            input_data = inputs
        x = torch.from_numpy(input_data).float()
        outputs = self.forward(Variable(x))
        # value, index = torch.max(outputs.data, 0, keepdim=True)
        pred = outputs.data.max(0)[1]
        return int(pred)

    def save_model_parameters(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model_parameters(self, filepath):
        self.load_state_dict(torch.load(filepath))

    def fit(self, train_input, train_target):
        current_loss = 0
        all_losses = []

        # first normalize all the data
        if self.scale:
            input_data = self.scaler.fit_transform(train_input)
        else:
            input_data = train_input

        iteration = 0
        for e in range(self.epochs):  # loop over the dataset multiple times
            for i in range(self.steps):
                # zero the parameter gradients
                self.optimizer.zero_grad()

                inputs, labels = self._train_input_fn(input_data, train_target)
                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                loss = loss.data[0]

                self.optimizer.step()
                current_loss += loss

                # print statistics
                if i % self.print_every == 0:
                    print('%d %d%% %.4f' % (iteration, iteration / float(self.steps * self.epochs) * 100.0, loss))

                # Add current loss avg to list of losses
                if i % self.plot_every == 0:
                    all_losses.append(current_loss / self.plot_every)
                    current_loss = 0
                iteration += 1
        return all_losses
