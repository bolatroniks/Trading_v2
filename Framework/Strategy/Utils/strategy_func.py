from matplotlib import pyplot as plt

import numpy as np

from Config.const_and_paths import NEUTRAL_SIGNAL


def plot_signals (ds, bMultiple = False, bSave=False, label = '', plot_filename=''):
    if not bMultiple:
        fig = plt.figure ()
    plt.title ('Signals')
    axes = plt.gca()
    ax = axes.twinx ()
    axes.plot(ds.p_df.Predictions, color='red')
    ax.plot(ds.f_df['Close_'+ds.timeframe])

    if not bMultiple:
        plt.show ()
        if bSave:
            fig.savefig(plot_filename)

    return fig


def compute_hit_miss_array (ds):
    hit_miss = (1*(ds.l_df.Labels == ds.p_df.Predictions) *
                (ds.p_df.Predictions != NEUTRAL_SIGNAL) - 0.5 *
                (ds.p_df.Predictions != NEUTRAL_SIGNAL) ) * 2

    return hit_miss


def plot_pnl (ds,
              bMultiple = False, bSave=False,
              label = '', plot_filename=''):
    if not bMultiple:
        fig = plt.figure ()
    plt.title ('PnL')
    #pnl = np.array([1 if label == pred else -1 for label, pred in zip (ds.l_df.Labels, ds.p_df.Predictions)])
    #plt.plot (np.cumsum(pnl))

    hit_miss_array = compute_hit_miss_array(ds)
    pnl = np.maximum(hit_miss_array, 0) * ds.target_multiple + np.minimum (hit_miss_array, 0) * 1

    plt.plot(np.cumsum(pnl), label = label)
    if not bMultiple:
        plt.show ()
        if bSave:
            fig.savefig(plot_filename)

    return fig


def plot_histogram (ds, bMultiple = False, bSave=False, label = '', plot_filename=''):
    if not bMultiple:
        fig = plt.figure ()
    plt.title ('Hit ratio - Histogram')
    #plt.hist(ds.l_df.Labels[ds.p_df.Predictions != 0] * ds.p_df.Predictions[ds.p_df.Predictions != 0], bins=5)
    hit_miss_array = compute_hit_miss_array(ds)
    plt.hist(hit_miss_array[hit_miss_array!=0], bins=5)
    if not bMultiple:
        plt.show ()
        if bSave:
            fig.savefig(plot_filename)

    return fig