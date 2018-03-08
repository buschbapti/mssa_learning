#!/usr/bin/env python
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def postural_distance(post1, post2):
    nb_dof = int(len(post1) / 3)
    post_dist = 0
    for d in range(nb_dof):
        p1 = [post1[3*d], post1[3*d+1], post1[3*d+2]]
        p2 = [post2[3*d], post2[3*d+1], post2[3*d+2]]
        post_dist += euclidean(p1, p2)
    return post_dist


def dynamic_time_warping(dataset):
    max_index = np.argmax([len(x) for x in dataset])
    ref = dataset[max_index]
    warped_series = []
    for i, timeserie in enumerate(dataset):
        if i != max_index:
            distance, path = fastdtw(ref, timeserie, dist=postural_distance)
            ws = []
            for p in path:
                ws.append(timeserie[p[1]])
            warped_series.append(ws)
        else:
            warped_series.append(ref)
    return warped_series


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
            SB = t[:3]
            SM = t[3:6]
            HL = t[36:39]
            HR = t[48:51]
            base_to_cam = base_transformation(SB, HL, HR, SM)
            for d in range(1, 25):
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