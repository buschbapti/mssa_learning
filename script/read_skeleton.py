#!/usr/bin/env python
"""NTU RGB+D Action Recognition Dataset to rpl converter

Usage:
    read_skeleton (-f FILE | <p> <r> <a>) [options]
    read_skeleton (-h | --help)

Arguments:
    p                    Id of the performer (1 to 20)
    r                    Number of the repetition (1 or 2)
    a                    Id of the action (1 to 60)

Options:
    -h --help                         Show the help menu.
    -f=FILE --file=FILE               Input file path
    -d=OUTDIR --output_dir=OUTDIR     Output directory path [default: /tmp/dataset]
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


def convert_skeleton_file(datafiles, output_filename, output_dir, kinect_id=None):
    joint_names = ['SpineBase', 'SpineMid', 'Neck', 'Head', 'ShoulderLeft', 'ElbowLeft',
                   'WristLeft', 'HandLeft', 'ShoulderRight', 'ElbowRight', 'WristRight',
                   'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', 'HipRight',
                   'KneeRight', 'AnkleRight', 'FootRight', 'SpineShoulder', 'HandTipLeft',
                   'ThumbLeft', 'HandTipRight', 'ThumbRight']

    frames = OrderedDict()
    for k_id, datafile in enumerate(datafiles):
        if kinect_id is not None:
            frames[kinect_id] = []
        else:
            frames[str(k_id + 1)] = []
        nb_frames = int(datafile.readline().strip())
        for i in range(nb_frames):
            frame = []
            nb_bodies = int(datafile.readline().strip())
            # for now only read first body
            for j in range(nb_bodies):
                line = datafile.readline().strip().split()
                trackid = int(line[0])
                nb_joints = int(datafile.readline().strip())
                frame.append({})
                for k in range(nb_joints):
                    joint_line = datafile.readline().strip().split()
                    joint = {}
                    joint["bodyTrackingId"] = {"bodyTrackingId": trackid}
                    pos = [float(x) for x in joint_line[:3]]
                    rot = [float(x) for x in joint_line[-5:-1]]
                    joint["position"] = {'x': pos[0], 'y': pos[1], 'z': pos[2]}
                    joint["orientation"] = {'w': rot[0], 'x': rot[1], 'y': rot[2], 'z': rot[3]}
                    # joint["depth_corrdinates"] = [float(x) for x in joint_line[3:5]]
                    # joint["image_corrdinates"] = [float(x) for x in joint_line[5:7]]
                    joint["trackingState"] = {"trackingState": int(joint_line[-1])}
                    joint["bodyCount"] = {"bodyCount": nb_bodies}
                    frame[-1][joint_names[k]] = joint
            if kinect_id is not None:
                frames[kinect_id].append(frame)
            else:
                frames[str(k_id + 1)].append(frame)
        datafile.close()

    # write file
    file = open(join(output_dir, "rpl", output_filename + ".rpl"), 'w')
    h5_dataset =  h5py.File(join(output_dir, "h5", "poses.h5"), 'a')
    group = h5_dataset.create_group(output_filename)

    dt = 1/30.
    kinects = frames.keys()

    min_lenght = min([len(frames[x]) for x in kinects])
    data = {}
    for k in kinects:
        data[k] = []

    for i in range(min_lenght):
        for k in kinects:
            vect = []
            f = frames[k][i][0]
            dict_f = {str(k): f}
            line = json.dumps(dict_f) + '@' + str(i*dt) + '\n'
            for j in joint_names:
                vect.append(f[j]["position"]['x'])
                vect.append(f[j]["position"]['y'])
                vect.append(f[j]["position"]['z'])
            file.write(line)
            data[k].append(vect)

    for k in kinects:
        group.create_dataset(k, data=data[k])

    file.close()
    h5_dataset.close()


if __name__ == '__main__':
    args = docopt(__doc__)
    schema = Schema({
        "--help": bool,
        "<p>": And(Use(int), lambda n: 1 < n < 40,
                           error="<p> should be integer 1 < n < 40."),
        "<r>": And(Use(int), lambda n: 1 < n < 2,
                           error="<r> should be integer 1 < n < 2."),
        "<a>": And(Use(int), lambda n: 1 < n < 60,
                           error="<a> should be integer 1 < n < 60."),
        "--output_file": str
        })

    with open("./missing_skeletons.json") as datafile:
        missing_list = json.load(datafile)

    if args["--file"]:
        filename = args["--file"]
        output_filename = filename[-29:-9]
        datafiles = [filename]
        kinect_id = args["--file"][-22]
        if output_filename in missing_list:
            print("Files of the dataset is corrupted do not use this file")
            sys.exit(1)
    else:
        pid = args["<p>"].zfill(3)
        rid = args["<r>"].zfill(3)
        aid = args["<a>"].zfill(3)
        datadir = "./nturgb+d_skeletons/"
        datafiles = []
        for i in range(3):
            filename = 'C00' + str(i + 1) + 'P' + pid + 'R' + rid + 'A' + aid
            output_filename = 'P' + pid + 'R' + rid + 'A' + aid
            if filename not in missing_list:
                datafiles.append(open(join(datadir, filename + '.skeleton'), 'r'))
        kinect_id = None

    if not isdir(args["--output_dir"]):
        makedirs(args["--output_dir"])
        makedirs(join(args["--output_dir"], "rpl"))
        makedirs(join(args["--output_dir"], "h5"))

    print("Converting " + output_filename)
    convert_skeleton_file(datafiles, output_filename, args["--output_dir"], kinect_id)
