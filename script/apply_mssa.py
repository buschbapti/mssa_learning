#!/usr/bin/env python
import h5py
import numpy as np
from mssa_learning.mssa import MSSA
from mssa_learning.tools.preprocessing import *
import os
from os.path import join
import progressbar


def main():
    script_path = os.path.dirname(os.path.realpath(__file__))
    filelist = ["P" + str(i).zfill(3) + "R" + str(j).zfill(3) + "A" + str(k).zfill(3) for i in range(1, 41) for j in [1, 2] for k in range(1, 61)]
    mssa = MSSA()

    bar = progressbar.ProgressBar()
    for recordings in bar(filelist):
        h5_pc = h5py.File(join(script_path, "..", "data", "dataset_converted", "h5", "pc_poses.h5"), 'a')
        if recordings not in h5_pc:
            group = h5_pc.create_group(recordings)
        for camera in ["1", "2", "3"]:
            h5_data = h5py.File(join(script_path, "..", "data", "dataset_converted", "h5", "poses.h5"))
            # only write the file if the original recordings are present
            if camera in h5_data[recordings] and ("PC" + camera) not in h5_pc[recordings]:
                data = h5_data[recordings][camera]
                rebased_data = rebase([data])
                transposed_data = transpose(rebased_data)
                pc, eig_vec, eig_val = mssa.compute_principal_components(transposed_data[0])
                h5_pc[recordings].create_dataset("PC" + camera, data=pc[:,:100])
            h5_data.close()
        h5_pc.close()

    bar = progressbar.ProgressBar()
    for recordings in bar(filelist):
        h5_pc = h5py.File(join(script_path, "..", "data", "dataset_converted", "h5", "pc_kalman.h5"), 'a')
        if recordings not in h5_pc:
            group = h5_pc.create_group(recordings)
        h5_data = h5py.File(join(script_path, "..", "data", "dataset_converted", "h5", "kalman.h5"))
        # only write the file if the original recordings are present
        if len(h5_data[recordings]["thetas"]) > 0 and "PC_thetas" not in h5_pc[recordings]:
            data = h5_data[recordings]["thetas"]
            transposed_data = transpose([data])
            pc, eig_vec, eig_val = mssa.compute_principal_components(transposed_data[0])
            h5_pc[recordings].create_dataset("PC_thetas", data=pc[:,:100])
        h5_data.close()
        h5_pc.close()


if __name__ == '__main__':
    main()