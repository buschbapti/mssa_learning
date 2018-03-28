#!/usr/bin/env python
import h5py
import numpy as np
import os
from os.path import join
import progressbar
import rospy
from glob import glob
from threading import RLock
from mssa_learning.tools.mock import Mock
from mssa_learning.tools.command_kalman import CommandKalman
from mssa_learning.tools.model import get_theta_names
from kombos_msgs.msg import kalmanState


class ApplyKalman(object):
    def __init__(self):
        self.sub = rospy.Subscriber("/kalmanState", kalmanState, self.handle_kalman_msgs)
        script_path = os.path.dirname(os.path.realpath(__file__))
        # list all files in the dataset
        data_path = join(script_path, "..", "data", "dataset_converted", "rpl")
        self.files = self.listdir_fullpath(data_path)
        self.record = []
        self.saved_records = []
        self.lock = RLock()
        self.h5_data_path = join(script_path, "..", "data", "dataset_converted", "h5", "kalman.h5")

        self.mock = Mock()
        self.kalman = CommandKalman()

        self.check_already_saved_records()

    def check_already_saved_records(self):
        h5_data = h5py.File(self.h5_data_path)
        self.saved_records = h5_data.keys()

    @staticmethod
    def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]

    def handle_kalman_msgs(self, msg):
        theta_values = [msg.value[msg.name.index(x)] for x in get_theta_names()]
        with self.lock:
            self.record.append(theta_values)

    def save_record(self, group_entry):
        h5_data = h5py.File(self.h5_data_path, 'a')
        group = h5_data.create_group(group_entry)
        group.create_dataset("thetas", data=self.record)
        h5_data.close()
        with self.lock:
            self.record = []

    def run(self, erase=False):
        bar = progressbar.ProgressBar()
        for f in bar(self.files):
            group_entry = f[-16:-4]
            if not group_entry in self.saved_records:
                # start kalman
                self.kalman.start_record()
                rospy.sleep(0.5)
                # play file
                self.mock.play(f)
                # stop kalman
                rospy.sleep(0.5)
                self.kalman.stop_record()
                # save recording
                self.save_record(group_entry)
            if rospy.is_shutdown():
                break
        return 0


if __name__ == '__main__':
    rospy.init_node("apply_kalman")
    kalman = ApplyKalman()
    kalman.run(False)