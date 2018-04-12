#!/usr/bin/env python
import h5py
from os.path import join
import numpy as np
import math
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


class H5Dataset(Dataset):
    def __init__(self, h5_filepath, window_size=30, normalize=True, nb_components=None, nb_labels=None, transform=None):
        self.h5_data = h5py.File(h5_filepath)
        self.indexed_list = []
        self.indexed_labels = []
        self.window_size = window_size
        self.normalize = normalize
        self.nb_components = nb_components
        self.nb_labels = nb_labels
        self.transform = transform
        self.index_data(window_size)

    def index_data(self, window_size):
        for key in self.h5_data:
            for rec in self.h5_data[key]:
                if len(self.h5_data[key][rec]) > 0:
                    if not math.isnan(self.h5_data[key][rec][0][0]):
                        label = int(key[-3:]) - 1
                        if not (self.nb_labels and label >= self.nb_labels):
                            for i in range(len(self.h5_data[key][rec]) - window_size):
                                self.indexed_list.append([key, rec, i, i+window_size])
                                self.indexed_labels.append(label)
        if not self.nb_labels:
            self.nb_labels = max(self.indexed_labels) + 1
        if not self.nb_components:
            indexes = self.indexed_list[0]
            self.nb_components = len(self.h5_data[indexes[0]][indexes[1]][0])

    def __getitem__(self, index):
        indexes = self.indexed_list[index]
        # extract data from h5
        data = self.h5_data[indexes[0]][indexes[1]][indexes[2]:indexes[3], :self.nb_components]
        label = self.indexed_labels[index]
        sample = (data, label)
        if self.transform:
            sample =  self.transform(sample)
        return sample

    def __len__(self):
        return len(self.indexed_list)


class H5Loader(object):
    def __init__(self, h5_filepath, window_size=30, normalize=True, batch_size=200, use_gpu=False, nb_components=None, nb_labels=None, transform=None):
        self.dataset = H5Dataset(h5_filepath, window_size, normalize=normalize, nb_components=nb_components, nb_labels=nb_labels, transform=transform)
        self.indices = list(range(len(self.dataset)))
        self.max_target = self.dataset.nb_labels - 1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.nb_labels = self.dataset.nb_labels
        self.nb_components = self.dataset.nb_components
        
    def extract_training_base(self):
        idx = self.indices
        np.random.shuffle(idx)
        nb_samples = int(9*len(idx)/(10*(self.max_target + 1)))
        samples_per_class = np.zeros(self.max_target + 1)
        max_samples = np.ones(self.max_target + 1) * nb_samples
        sub_targets = self.dataset.indexed_labels

        i = 0
        train_idx = []
        test_idx = []
        while not np.equal(samples_per_class, max_samples).all() and i<len(sub_targets):
            if samples_per_class[sub_targets[idx[i]]] < nb_samples:
                train_idx.append(idx[i])
                samples_per_class[sub_targets[idx[i]]] += 1
            else:
                test_idx.append(idx[i])
            i += 1

        for j in range(i, len(sub_targets)):
            test_idx.append(idx[j])

        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        if self.use_gpu:
            train_data = DataLoader(self.dataset, sampler=train_idx, drop_last=True, batch_size=self.batch_size, num_workers=1, pin_memory=True)
            test_data = DataLoader(self.dataset, sampler=test_idx, drop_last=True, batch_size=self.batch_size, num_workers=1, pin_memory=True)
        else:
            train_data = DataLoader(self.dataset, sampler=train_idx, drop_last=True, batch_size=self.batch_size, num_workers=1)
            test_data = DataLoader(self.dataset, sampler=test_idx, drop_last=True, batch_size=self.batch_size, num_workers=1)
        return train_data, test_data
