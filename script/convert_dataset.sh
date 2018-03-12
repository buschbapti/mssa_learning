#!/bin/bash
for p in `seq 1 40`;
do
	for r in `seq 1 2`;
	do
		for a in `seq 1 60`;
		do
			python -W ignore read_skeleton.py $p $r $a -d "../data/dataset_converted"
		done
	done
done