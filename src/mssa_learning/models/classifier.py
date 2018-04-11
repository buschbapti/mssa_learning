import torch.nn
import numpy as np
from sklearn import preprocessing
import progressbar
import abc
from os.path import join
from os.path import isdir
from os import makedirs
import json


class Classifier(torch.nn.Module):
    """Abstract Class to represent a generic classifier"""
    def __init__(self, input_size, hidden_size, output_size, network_type, num_layers=1, batch_size=200, steps=100000, epochs=2, scale=False, save_folder=None):
        super(Classifier, self).__init__()
        __metaclass__ = abc.ABCMeta
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.use_gpu = False
        self.num_layers = num_layers
        self.network_type = network_type

        self.steps = steps
        self.epochs = epochs
        self.print_every = 5000
        self.save_folder = save_folder

        self.scalers = []
        for d in range(input_size):
            self.scalers.append(preprocessing.MinMaxScaler())
        self.scale = scale

    @abc.abstractmethod
    def _forward(self, input):
        """
        Apply the forward method of the neural network to the input data

        Keywoard arguments:
        input -- The single vector of features in entry of the network

        Return arguments:
        output -- Value returned by the last layer of the network
        """
        raise NotImplementedError()
        return 0

    @abc.abstractmethod
    def predict(self, inputs):
        """
        Predict the label given input variable using a forward pass and get the
        largest index

        Keywoard arguments:
        inputs -- The set of features to predict

        Return:
        predicted -- The set of predicted categories
        """
        raise NotImplementedError()
        return 0

    @abc.abstractmethod
    def fit(self, train_set, test_set=None):
        """
        Fit the network to the train set in input
        
        Keywoard arguments:
        train_set -- Tuple containing the features to train on
        and their corresponding labels
        test_set -- Optional, tuple containing the features to evaluate
        the learned model on and their corresponding labels
        """
        raise NotImplementedError()

    def evaluate(self, test_set):
        """
        Evaluate the network on the inputs and their corresponding targets
        
        Keywoard arguments:
        test_set -- Tuple containing the features to evaluate
        the learned model on and their corresponding labels

        Return:
        success_rate -- Pourcentage of correct predictions 
        """
        predicted = self.predict(test_set[0])
        nb_errors = 0
        nb_elem = len(test_set[0])
        for i, p in enumerate(predicted):
            if p != test_set[1][i]:
                nb_errors += 1
        success_rate = (nb_elem - nb_errors) / float(nb_elem)
        return success_rate

    def _fit_normalizer(self, input_data):
        """
        Fit the MinMaxScaler normalizer on the input_data
        
        Keywoard arguments:
        input_data -- Data to fit the normalizer on 
        """
        for d in range(self.input_size):
            self.scalers[d] = self.scalers[d].fit(input_data[:, :, d])

    def _normalize_data(self, input_data):
        """
        Apply the fitted MinMaxScaler normalizer on the input_data
        
        Keywoard arguments:
        input_data -- Data to normalize

        Return:
        scaled_data -- Normalized data
        """
        scaled_input = np.zeros((len(input_data), self.seq_len, self.input_size))
        for d in range(self.input_size):
            scaled_input[:, :, d] = self.scalers[d].transform(input_data[:, :, d])
        return scaled_input

    def _save_results(self):
        """
        Save the learning curve in a json file
        """
        if self.save_folder is not None:
            if not isdir(self.save_folder):
                makedirs(self.save_folder)
            filename = (self.network_type + "_" + str(self.output_size) + "c_" +
                        str(self.seq_len) + "t_" + str(self.hidden_size) + "h_" +
                        str(self.num_layers) + "l_" + str(self.batch_size) + "b.json")
            save_data = {}
            save_data["loss"] = self.cumulative_loss
            save_data["evaluation"] = self.cumulative_eval
            with open(join(self.save_folder, filename), 'w') as savefile:
                json.dump(save_data, savefile)

    def _save_model(self):
        """
        Save the learned model in a picke file
        """
        return 0

    def _score(self, iteration, loss, test_set=None):
        """
        Score the learned model and print learning progress
        
        Keywoard arguments:
        iteration -- Current iteration
        loss -- Cumulative loss since last scoring
        test_set -- Optional, tuple containing the evaluation set
        """
        print('%d %d%% %.4f' % (iteration, iteration / float(self.steps * self.epochs) * 100.0, loss))
        self.cumulative_loss.append(float(loss))
        if test_set is not None:
            evaluation = self.evaluate(test_set)
            print('success rate: %f' % (evaluation))
            self.cumulative_eval.append(float(evaluation))
