#!/usr/bin/env python
import h5py
from os.path import join
import numpy as np
import math


def h5_to_numpy(h5_data, window_size=30, nb_components=None):
    features = []
    labels = []

    for key in h5_data:
        for rec in h5_data[key]:
            if len(h5_data[key][rec]) > 0:
                if not math.isnan(h5_data[key][rec][0][0]):
                    label = int(key[-3:]) - 1
                    for i in range(len(h5_data[key][rec]) - window_size):
                        labels.append(label)
                        if nb_components is None:
                            features.append(h5_data[key][rec][i:i+window_size])
                        else:
                            features.append(h5_data[key][rec][i:i+window_size, :nb_components])
    return np.array(features), np.array(labels)


def create_training_base(features, targets, max_target=None):
    if max_target is None:
        max_target = max(targets)
        sub_features = features
        sub_targets = targets
    else:
        sub_features = []
        sub_targets = []
        for i, t in enumerate(targets):
            if t <= max_target:
                sub_features.append(features[i])
                sub_targets.append(t)

    idx = np.arange(len(sub_targets))
    np.random.shuffle(idx)
    nb_samples = int(9*len(idx)/(10*(max_target + 1)))
    samples_per_class = np.zeros(max_target + 1)
    max_samples = np.ones(max_target + 1) * nb_samples

    i = 0
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    while not np.equal(samples_per_class, max_samples).all() and i<len(sub_targets):
        if samples_per_class[sub_targets[idx[i]]] < nb_samples:
            train_x.append(sub_features[idx[i]])
            train_y.append(sub_targets[idx[i]])
            samples_per_class[sub_targets[idx[i]]] += 1
        else:
            test_x.append(sub_features[idx[i]])
            test_y.append(sub_targets[idx[i]])
        i += 1

    for j in range(i, len(sub_targets)):
        test_x.append(sub_features[idx[j]])
        test_y.append(sub_targets[idx[j]])

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)