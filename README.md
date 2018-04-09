# Multivariate Singular Spectrum Analysis

This repository contains code to apply Multivariate Singular Spectrum Analysis (MSSA) on Human Skeleton data in order to filter the signal.
The method is described in the notebook [mssa_tutorial](notebooks/mssa_tutorial.ipynb) which provide an example of the technique over a simple signal in 2 dimensions.

The second notebook [mssa_dataset_analysis](notebooks/mssa_dataset_analysis.ipynb) contains examples to apply the technique to skeleton data.
For now, the notebook uses data from the [NTU RGB+D Action Recognition Dataset](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp).
This dataset require authorization to be downloaded. The script folder contains python script to convert the data in h5 format.

You can also apply the technique on your own data if you respect the structure of the input data.
The data should be a 2d-array with dimensions T X D where T is the number of time frames of the time series and D the number of dimensions.
For skeleton data, each dimensions d contains positions (x, y or z) in space of a body point (e.g. shoulder, head, ...).
Ultimately, you can also apply MSSA on any multivariate time series by providing data in the same format.
