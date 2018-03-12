#!/usr/bin/env python
import numpy as np
from mssa_learning.preprocessing import get_joint_names, extract_position


def calculate_lengths(timeserie):
    lengths = {}
    lengths["rhumerus"] = []
    lengths["lhumerus"] = []
    lengths["rradius"] = []
    lengths["lradius"] = []
    lengths["rmain"] = []
    lengths["lmain"] = []
    lengths["rclavicule"] = []
    lengths["lclavicule"] = []
    lengths["thorax"] = []
    lengths["abdomen"] = []
    lengths["rpelvis"] = []
    lengths["lpelvis"] = []
    lengths["rfemur"] = []
    lengths["lfemur"] = []
    lengths["rtibia"] = []
    lengths["ltibia"] = []
    lengths["cou"] = []

    for posture in timeserie:
        lengths["rhumerus"].append(np.linalg.norm(extract_position(posture, "ElbowRight", True) - extract_position(posture, "ShoulderRight", True)))
        lengths["lhumerus"].append(np.linalg.norm(extract_position(posture, "ElbowLeft", True) - extract_position(posture, "ShoulderLeft", True)))
        lengths["rradius"].append(np.linalg.norm(extract_position(posture, "WristRight", True) - extract_position(posture, "ElbowRight", True)))
        lengths["lradius"].append(np.linalg.norm(extract_position(posture, "WristLeft", True) - extract_position(posture, "ElbowLeft", True)))
        lengths["rmain"].append(np.linalg.norm(extract_position(posture, "HandRight", True) - extract_position(posture, "WristRight", True)))
        lengths["lmain"].append(np.linalg.norm(extract_position(posture, "HandLeft", True) - extract_position(posture, "WristLeft", True)))
        lengths["rclavicule"].append(np.linalg.norm(extract_position(posture, "SpineShoulder", True) - extract_position(posture, "ShoulderRight", True)))
        lengths["lclavicule"].append(np.linalg.norm(extract_position(posture, "SpineShoulder", True) - extract_position(posture, "ShoulderLeft", True)))
        lengths["thorax"].append(np.linalg.norm(extract_position(posture, "SpineMid", True) - extract_position(posture, "SpineShoulder", True)))
        lengths["abdomen"].append(np.linalg.norm(extract_position(posture, "SpineMid", True)))
        lengths["rpelvis"].append(np.linalg.norm(extract_position(posture, "HipRight", True)))
        lengths["lpelvis"].append(np.linalg.norm(extract_position(posture, "HipLeft", True)))
        lengths["rfemur"].append(np.linalg.norm(extract_position(posture, "KneeRight", True) - extract_position(posture, "HipRight", True)))
        lengths["lfemur"].append(np.linalg.norm(extract_position(posture, "KneeLeft", True) - extract_position(posture, "HipLeft", True)))
        lengths["rtibia"].append(np.linalg.norm(extract_position(posture, "AnkleRight", True) - extract_position(posture, "KneeRight", True)))
        lengths["ltibia"].append(np.linalg.norm(extract_position(posture, "AnkleLeft", True) - extract_position(posture, "KneeLeft", True)))
        lengths["cou"].append(np.linalg.norm(extract_position(posture, "Head", True) - extract_position(posture, "SpineShoulder", True)))
    return lengths
