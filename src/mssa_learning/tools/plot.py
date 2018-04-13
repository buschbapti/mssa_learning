#!/usr/bin/env python
import plotly.graph_objs as go
from plotly import tools
import numpy as np


def plot_explained_variance(eig_vals, nb_components):
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in eig_vals]
    cum_var_exp = np.cumsum(var_exp)

    trace1 = go.Bar(x=['PC %s' % i for i in range(1, nb_components + 1)], y=var_exp, showlegend=False)

    trace2 = go.Scatter(x=['PC %s' % i for i in range(1,nb_components + 1)], y=cum_var_exp, name='cumulative explained variance')

    data = go.Data([trace1, trace2])

    layout = go.Layout(yaxis=go.YAxis(title='Explained variance in percent'), title='Explained variance by different principal components')

    fig = go.Figure(data=data, layout=layout)
    return fig


def plot_timeseries(data, rs_data=None):
    D = len(data)
    N = len(data[0])

    fig = tools.make_subplots(rows=25, cols=3, print_grid=False)
    for d in range(int(D/3)):
        for i in range(3):
            trace = go.Scatter(x=np.arange(N), y=data[3*d+i])
            fig.append_trace(trace, d+1, i+1)
            # add the reconstructed signal if provided
            if rs_data is not None:
                rs_trace = go.Scatter(x=np.arange(N), y=rs_data[3*d+i])
                fig.append_trace(rs_trace, d+1, i+1)
    fig['layout'].update(height=1800, width=800, title='timeseries')
    return fig


def plot_principal_components(pcs, nb_components=10):
    fig = tools.make_subplots(rows=nb_components, cols=1, print_grid=False)
    for pc in pcs:
        for k in range(nb_components):
            data_tmp = pc[:, k]
            trace = go.Scatter(x=np.arange(len(data_tmp)), y=data_tmp)
            fig.append_trace(trace, k+1, 1)
        fig['layout'].update(height=1000, width=900, title='principal components')
    return fig
