#!/bin/bash
DATASET_DIR="../data/nturgb+d_skeletons"
for entry in $DATASET_DIR/*
do
	python -W ignore convert_skeleton.py -f $entry -d ../data/dataset_converted
done
