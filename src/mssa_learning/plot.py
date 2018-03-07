#!/usr/bin/env python
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
import numpy as np


def plot_timeseries(data):
    D = len(data)
    N = len(data[0])

    fig = tools.make_subplots(rows=25, cols=3)
    for d in range(int(D/3)):
        for i in range(3):
            trace = go.Scatter(
                x = np.arange(N),
                y = data[3*d+i]
            )
            fig.append_trace(trace, d+1, i+1)
    fig['layout'].update(height=1800, width=800, title='timeseries')
    return fig


def plot_principal_components(pcs, nb_components=10):
    fig = tools.make_subplots(rows=nb_components, cols=1)
    for pc in pcs:
        for k in range(nb_components):
            data_tmp = pc[:, k]
            trace = go.Scatter(
                x = np.arange(len(data_tmp)),
                y = data_tmp
            )
            fig.append_trace(trace, k+1, 1)
        fig['layout'].update(height=1000, width=900, title='principal components')
    return fig