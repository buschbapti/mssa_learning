#!/usr/bin/env python
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import progressbar


def get_joint_names(rebased=False):
    joint_names = ['SpineBase', 'SpineMid', 'Neck', 'Head', 'ShoulderLeft', 'ElbowLeft',
                   'WristLeft', 'HandLeft', 'ShoulderRight', 'ElbowRight', 'WristRight',
                   'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', 'HipRight',
                   'KneeRight', 'AnkleRight', 'FootRight', 'SpineShoulder', 'HandTipLeft',
                   'ThumbLeft', 'HandTipRight', 'ThumbRight']
    if rebased:
        joint_names.remove("SpineBase")
    return joint_names


def extract_position(posture, joint_name, rebased=False):
    joint_names = get_joint_names(rebased)
    idx = joint_names.index(joint_name)
    return np.array(posture)[3*idx:3*idx+3]


def postural_distance(post1, post2):
    nb_dof = int(len(post1) / 3)
    post_dist = 0
    for d in range(nb_dof):
        p1 = [post1[3*d], post1[3*d+1], post1[3*d+2]]
        p2 = [post2[3*d], post2[3*d+1], post2[3*d+2]]
        post_dist += euclidean(p1, p2)
    return post_dist

def principal_components_distance(pc1, pc2):
    return np.linalg.norm(pc1 - pc2)

def dynamic_time_warping(dataset, dist):
    max_index = np.argmax([len(x) for x in dataset])
    ref = dataset[max_index]
    warped_series = []
    bar = progressbar.ProgressBar()
    for i in bar(range(len(dataset))):
        timeserie = dataset[i]
        if i != max_index:
            distance, path = fastdtw(ref, timeserie, dist=dist)
            ws = []
            for p in path:
                ws.append(timeserie[p[1]])
            warped_series.append(np.array(ws))
        else:
            warped_series.append(ref)
    return warped_series, len(dataset[max_index])


def base_transformation(pb, px1, px2, py):
    tmp = (px2 - px1) / np.linalg.norm(px2 - px1)
    vy = (py - pb) / np.linalg.norm(py - pb)
    vx = tmp - ((np.dot(vy, tmp)) * vy)
    vx = vx / np.linalg.norm(vx)
    vz = np.cross(vx, vy)
    return [[vx[0], vx[1], vx[2], -vx[0]*pb[0]-vx[1]*pb[1]-vx[2]*pb[2]],
            [vy[0], vy[1], vy[2], -vy[0]*pb[0]-vy[1]*pb[1]-vy[2]*pb[2]],
            [vz[0], vz[1], vz[2], -vz[0]*pb[0]-vz[1]*pb[1]-vz[2]*pb[2]],
            [0, 0, 0, 1]]


def rebase(dataset):
    new_dataset = []
    for timeserie in dataset:
        new_timeserie = []
        for t in timeserie:
            new_time = []
            SB = extract_position(t, "SpineBase")
            SM = extract_position(t, "SpineMid")
            HL = extract_position(t, "HipLeft")
            HR = extract_position(t, "HipRight")
            base_to_cam = base_transformation(SB, HL, HR, SM)
            for d in range(1, len(get_joint_names())):
                p = [t[3*d], t[3*d+1], t[3*d+2], 1]
                new_p = np.dot(base_to_cam, p).tolist()
                new_time += new_p[:-1]
            new_timeserie.append(new_time)
        new_dataset.append(new_timeserie)
    return new_dataset


def transpose(dataset):
    new_dataset = []
    for timeserie in dataset:
        new_dataset.append(np.transpose(timeserie))
    return new_dataset
