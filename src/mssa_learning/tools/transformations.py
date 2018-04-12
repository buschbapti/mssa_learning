#!/usr/bin/env python
from mssa_learning.tools.preprocessing import extract_position
from mssa_learning.tools.preprocessing import base_transformation
from mssa_learning.tools.preprocessing import get_joint_names
import numpy as np
import torch


class Rebase(object):
    def __call__(self, sample):
        rebased_sample = []
        for t in sample[0]:
            new_time = []
            SB = extract_position(t, "SpineBase")
            SM = extract_position(t, "SpineMid")
            HL = extract_position(t, "HipLeft")
            HR = extract_position(t, "HipRight")
            base_to_cam = base_transformation(SB, HL, HR, SM)
            for d in range(len(get_joint_names())):
                p = [t[3*d], t[3*d+1], t[3*d+2], 1]
                new_p = np.dot(base_to_cam, p).tolist()
                new_time += new_p[:-1]
            rebased_sample.append(new_time)
        return rebased_sample, sample[1]


class Normalize(object):
    def __init__(self, scalers):
        self.scalers = scalers

    def __call__(self, sample):
        scaled_sample = np.zeros(len(sample[0]), len(sample[0][0]))
        for d in range(len(sample[0][0])):
            scaled_sample[:, d] = self.scalers[d].transform(sample[0][:, d])
        return scaled_sample, sample[1]


class ToTensor(object):
    """docstring for ToTensor"""
    def __call__(self, sample):
        data_tensor = torch.from_numpy(np.array(sample[0])).float()
        label_tensor = torch.from_numpy(np.array([sample[1]]))
        return data_tensor, label_tensor