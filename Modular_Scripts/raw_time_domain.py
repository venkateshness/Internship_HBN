from plotly.graph_objs import Layout, YAxis, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
from plotly import tools
from plotly import graph_objects
import chart_studio.plotly as py
import matplotlib.pyplot as plt
import mne


def time_domain_plot(raw):
    n_channels = 20
    start, stop = raw.time_as_index([0, 5])
    picks = mne.pick_channels(
        raw.ch_names, include=raw.ch_names[:n_channels], exclude=[])

    data, times = raw[picks[:n_channels], start:stop]
    ch_names = [raw.info['ch_names'][p] for p in picks[:n_channels]]
    # ch_names

    step = 1. / n_channels
    kwargs = dict(domain=[1 - step, 1], showticklabels=False,
                  zeroline=False, showgrid=False)

    # create objects for layout and traces
    layout = Layout(yaxis=YAxis(kwargs), showlegend=False)
    traces = [Scatter(x=times, y=data.T[:, 0])]

    # loop over the channels
    for ii in range(1, n_channels):
        kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
        layout.update(
            {'yaxis%d' % (ii + 1): YAxis(kwargs), 'showlegend': False})
        traces.append(
            Scatter(x=times, y=data.T[:, ii], yaxis='y%d' % (ii + 1)))

    # add channel names using Annotations
    annotations = Annotations([Annotation(x=-0.06, y=0, xref='paper', yref='y%d' % (ii + 1),
                                          text=ch_name, font=Font(size=9), showarrow=False)
                               for ii, ch_name in enumerate(ch_names)])
    layout.update(annotations=annotations)

    # set the size of the figure and plot it
    layout.update(autosize=False, width=1000, height=600)
    fig = Figure(data=Data(traces), layout=layout)
    return fig
