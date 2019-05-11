#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 16:35:23 2018

@author: joanna
"""

import warnings
warnings.filterwarnings("ignore")

from matplotlib import cm
from matplotlib.dates import YearLocator, MonthLocator


import seaborn as sns

# get fed data
from Trading.Dataset.Dataset import *
from Trading.FeatureExtractors.Model.TimeSeries.ernst_chen import *

plt.style.use('ggplot')

ds = Dataset(ccy_pair='USD_ZAR', timeframe='D', from_time=2011, to_time=2015)
#ds.loadFeatures ()
ds.loadCandles ()
ds.computeFeatures (bComputeCandles=False, bComputeHighLowFeatures=False, bComputeIndicators=False, bComputeNormalizedRatios=False)
ds.f_df['hist_vol_1m_close'] = ds.f_df['Change'].rolling(window=int(22)).std() * (252.0/1.0) ** 0.5

ds.f_df.dropna (inplace = True)
ds.computeLabels (min_stop = 0.02)

df = ds.f_df
slow_timeframe = ds.timeframe

#df['trendlines_diff_10'] = df['no_standing_highs_10'] - df['no_standing_lows_10']

if False:
    b = np.array([halflife (ds.f_df.Close[i-252:i]) for i in range(252, len(ds.f_df))])
    ds.f_df['halflife'] = np.zeros(len(ds.f_df))
    ds.f_df['halflife'][252:] = b[:,1]
    
    ds.f_df.halflife.plot ()

cols = ['Change', 'hist_vol_1m_close']

X = np.reshape(ds.f_df[cols], (len(ds.f_df),len (cols)))
#X = df[cols]

#np.minimum(np.abs(df.halflife),252).plot ()

# gives us a quick visual inspection of the data
#msno.matrix(data)


# code adapted from http://hmmlearn.readthedocs.io
# for sklearn 18.1

#col = 'sret'
#select = data.ix[:].dropna()

#ft_cols = [f1, f2, f3, 'sret']
#X = select[ft_cols].values

if False:
    model = hmm.GaussianMixture(n_components=3, 
                            covariance_type="full", 
                            n_init=100, 
                            random_state=1).fit(X)
    
if False:
    model = hmm.GaussianHMM(n_components=3, 
                            covariance_type="diag").fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)


plt.plot(hidden_states)

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
#    print("var = ", np.diag(model.covariances_[i]))
    print()

sns.set(font_scale=1.25)
style_kwds = {'xtick.major.size': 3, 'ytick.major.size': 3,
               'legend.frameon': True}
sns.set_style('white', style_kwds)

fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True, figsize=(12,9))
colors = cm.rainbow(np.linspace(0, 1, model.n_components))

for i, (ax, color) in enumerate(zip(axs, colors)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot(ds.f_df.Close[mask],
                 ".-", c=color)
    ax.set_title("{0}th hidden state".format(i), fontsize=16, fontweight='demi')

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    sns.despine(offset=10)

plt.tight_layout()
fig.savefig('Hidden Markov (Mixture) Model_Regime Subplots.png')

sns.set(font_scale=1.5)
states = (pd.DataFrame(hidden_states, columns=['states'], index=ds.f_df.index)
          .join(ds.f_df, how='inner')
          .assign(mkt_cret=ds.f_df.Change.cumsum())
          .reset_index(drop=False))
print(states.head())

sns.set_style('white', style_kwds)
order = [0, 1, 2, 3, 4]
fg = sns.FacetGrid(data=states, hue='states', hue_order=order,
                   aspect=1.31, size=12)
fg.map(plt.scatter, 'Date', 'Close', alpha=0.8).add_legend()
sns.despine(offset=10)
fg.fig.suptitle('Historical SPY Regimes', fontsize=24, fontweight='demi')
#fg.savefig('Hidden Markov (Mixture) Model_SPY Regimes.png')