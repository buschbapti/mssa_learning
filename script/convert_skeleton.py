#!/usr/bin/env python
"""NTU RGB+D Action Recognition skeleton file to h5 converter

Usage:
    convert_skeleton (-f FILE | <s> <p> <r> <a>) [options]
    convert_skeleton (-h | --help)

Arguments:
    s    View ID (1 to 17)
    p    Performer ID (1 to 40)
    r    Number of the repetition (1 or 2)
    a    Action ID (1 to 60)

Options:
    -h --help                        Show the help menu.
    -f=FILE --file=FILE              Input file path
    -d=OUTDIR --output_dir=OUTDIR    Output directory path [default: /tmp/dataset]
"""
from os.path import join
import json
from docopt import docopt
from schema import Schema, And, Use, SchemaError
from os.path import isdir
from os import makedirs
from os.path import exists
import h5py
import sys
from collections import OrderedDict
import json
from mssa_learning.tools.preprocessing import get_joint_names
import os
import numpy as np


def convert_skeleton_file(datafiles, group_name, output_dir):
    h5file = join(output_dir, "h5", "poses2.h5")
    joint_names = get_joint_names()
    pose_data = {}
    rpl_data = {}
    dt = 1/30.
    for datafile in datafiles:
        ostream = open(datafile, 'r')
        kinect_id = datafile[-22]
        pose_data["camera" + kinect_id] = {}
        rpl_data["camera" + kinect_id] = {}
        nb_frames = int(ostream.readline().strip())
        for i in range(nb_frames):
            nb_bodies = int(ostream.readline().strip())
            for j in range(nb_bodies):
                line = ostream.readline().strip().split()
                trackid = line[0]
                nb_joints = int(ostream.readline().strip())
                if not "body" + trackid in pose_data["camera" + kinect_id].keys():
                    pose_data["camera" + kinect_id]["body" + trackid] = []
                if not "body" + trackid in rpl_data["camera" + kinect_id].keys():
                    rpl_data["camera" + kinect_id]["body" + trackid] = []
                pose_vect = []
                rpl_vect = []
                # time and id for rpl
                rpl_vect.append(i * dt)
                rpl_vect.append(trackid)
                for k in range(nb_joints):
                    joint_line = ostream.readline().strip().split()
                    pos = [float(x) for x in joint_line[:3]]
                    rot = [float(x) for x in joint_line[-5:-1]]
                    # only position for pose_data
                    pose_vect.append(pos[0])
                    pose_vect.append(pos[1])
                    pose_vect.append(pos[2])
                    # joint names for rpl
                    rpl_vect.append(joint_names[k])
                    # position for rpl
                    rpl_vect.append(pos[0])
                    rpl_vect.append(pos[1])
                    rpl_vect.append(pos[2])
                    # rotation for rpl
                    rpl_vect.append(rot[0])
                    rpl_vect.append(rot[1])
                    rpl_vect.append(rot[2])
                    rpl_vect.append(rot[3])
                    # tracking state for rpl
                    rpl_vect.append(joint_line[-1])
                pose_data["camera" + kinect_id]["body" + trackid].append(pose_vect)
                rpl_data["camera" + kinect_id]["body" + trackid].append(rpl_vect)
        ostream.close()
            
    h5data = h5py.File(h5file, 'a')
    if group_name in h5data:
        group = h5data[group_name]
    else:
        group = h5data.create_group(group_name)
    # write the data keeping the correct hierarchy
    for cam_id in rpl_data.keys():
        if cam_id in group:
            cam_group = group[cam_id]
        else:
            cam_group = group.create_group(cam_id)
        for body_id in rpl_data[cam_id].keys():
            if body_id in cam_group:
                body_group = cam_group[body_id]
                del body_group["poses"]
                del body_group["rpl"]
            else:
                body_group = cam_group.create_group(body_id)
            body_group.create_dataset("poses", data=pose_data[cam_id][body_id])
            body_group.create_dataset("rpl", data=rpl_data[cam_id][body_id])
    h5data.close()
    

    # for cam_id, camera_data in pose_data.iteritems():
    #     if not cam_id in group:
    #         cam_group = group.create_group(group_name)
    #     else:
    #         cam_group = group[cam_id]
    #     for body_id, body_data in camera_data.iteritems():
    #         group.create_dataset("poses_" + body_id + '_' + cam_id, data=body_data)
    #         group.create_dataset("rpl_" + body_id + '_' + cam_id, data=rpl_data[cam_id][body_id])
    # h5_dataset.close()

    # dt = 1/30.
    # kinects = frames.keys()

    # min_lenght = min([len(frames[x]) for x in kinects])
    # data = {}
    # for k in kinects:
    #     data[k] = []

    # for i in range(min_lenght):
    #     for k in kinects:
    #         vect = []
    #         f = frames[k][i][0]
    #         dict_f = {str(k): f}
    #         line = json.dumps(dict_f) + '@' + str(i*dt) + '\n'
    #         for j in joint_names:
    #             vect.append(f[j]["position"]['x'])
    #             vect.append(f[j]["position"]['y'])
    #             vect.append(f[j]["position"]['z'])
    #         file.write(line)
    #         data[k].append(vect)

    # for k in kinects:
    #     group.create_dataset(k, data=data[k])

    # file.close()
    


if __name__ == '__main__':
    args = docopt(__doc__)
    schema = Schema({
        "--help": bool,
        "<s>": And(Use(int), lambda n: 1 < n < 17,
                           error="<p> should be integer 1 < n < 17."),
        "<p>": And(Use(int), lambda n: 1 < n < 40,
                           error="<p> should be integer 1 < n < 40."),
        "<r>": And(Use(int), lambda n: 1 < n < 2,
                           error="<r> should be integer 1 < n < 2."),
        "<a>": And(Use(int), lambda n: 1 < n < 60,
                           error="<a> should be integer 1 < n < 60."),
        "--output_file": str
        })
    script_path = os.path.dirname(os.path.realpath(__file__))
    # load file ok known missing skeletons
    with open(join(script_path, "..", "config", "missing_skeletons.json")) as datafile:
        missing_list = json.load(datafile)
    if args["--file"]:
        filename = args["--file"]
        if filename in missing_list:
            print("Files {0:s} of the dataset is corrupted do not use this file".format(filename))
            sys.exit(1)
        if not exists(filename):
            print("File {0:s} does not exist".format(filename))
            sys.exit(1)
        datafiles = [args["--file"]]
        sid = filename[-28:-25]
        pid = filename[-20:17]
        rid = filename[-16:-13]
        aid = filename[-12:-9]
    else:
        sid = args["<s>"].zfill(3)
        pid = args["<p>"].zfill(3)
        rid = args["<r>"].zfill(3)
        aid = args["<a>"].zfill(3)
        datadir = join(script_path, "..", "data", "nturgb+d_skeletons")
        datafiles = []
        for i in range(3):
            filename = 'S' + sid + 'C00' + str(i + 1) + 'P' + pid + 'R' + rid + 'A' + aid
            if exists(filename) and filename not in missing_list:
                datafiles.append(join(datadir, filename + '.skeleton'))
        if not datafiles:
            print("Incorrect input parameters or all corresponding files are corrupted")
            sys.exit(1)
    group_name = 'S' + sid + 'P' + pid + 'R' + rid + 'A' + aid
    if not isdir(args["--output_dir"]):
        makedirs(args["--output_dir"])
        makedirs(join(args["--output_dir"], "h5"))
    print("Converting " + group_name)
    convert_skeleton_file(datafiles, group_name, args["--output_dir"])
