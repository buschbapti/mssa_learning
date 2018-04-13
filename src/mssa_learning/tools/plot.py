#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# def plot_explained_variance(eig_vals, nb_components):
#     tot = sum(eig_vals)
#     var_exp = [(i / tot)*100 for i in eig_vals]
#     cum_var_exp = np.cumsum(var_exp)

#     trace1 = go.Bar(x=['PC %s' % i for i in range(1, nb_components + 1)], y=var_exp, showlegend=False)

#     trace2 = go.Scatter(x=['PC %s' % i for i in range(1,nb_components + 1)], y=cum_var_exp, name='cumulative explained variance')

#     data = go.Data([trace1, trace2])

#     layout = go.Layout(yaxis=go.YAxis(title='Explained variance in percent'), title='Explained variance by different principal components')

#     fig = go.Figure(data=data, layout=layout)
#     return fig


# def plot_timeseries(data, rs_data=None):
#     D = len(data)
#     N = len(data[0])

#     fig = tools.make_subplots(rows=25, cols=3, print_grid=False)
#     for d in range(int(D/3)):
#         for i in range(3):
#             trace = go.Scatter(x=np.arange(N), y=data[3*d+i])
#             fig.append_trace(trace, d+1, i+1)
#             # add the reconstructed signal if provided
#             if rs_data is not None:
#                 rs_trace = go.Scatter(x=np.arange(N), y=rs_data[3*d+i])
#                 fig.append_trace(rs_trace, d+1, i+1)
#     fig['layout'].update(height=1800, width=800, title='timeseries')
#     return fig


# def plot_principal_components(pcs, nb_components=10):
#     fig = tools.make_subplots(rows=nb_components, cols=1, print_grid=False)
#     for pc in pcs:
#         for k in range(nb_components):
#             data_tmp = pc[:, k]
#             trace = go.Scatter(x=np.arange(len(data_tmp)), y=data_tmp)
#             fig.append_trace(trace, k+1, 1)
#         fig['layout'].update(height=1000, width=900, title='principal components')
#     return fig


def plot_data_distribution(data_set, nb_labels, title='Data distribution',):
    labels = np.zeros(nb_labels)
    for (_, target) in data_set:
        for l in target:
            labels[int(l)] += 1
    classes = np.arange(nb_labels)
    plt.bar(classes, labels)
    classes = [str(x) for x in classes]
    tick_marks = np.arange(len(classes)) + 0.5
    plt.title(title)
    plt.tight_layout()
    plt.xticks(tick_marks, classes)
    plt.ylabel('Number of examples')
    plt.xlabel('Label')

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = np.arange(len(cm))
    classes = [str(x) for x in classes]
    cm = cm.astype('float') / cm.sum(axis=1 )[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')