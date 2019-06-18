# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 22:01:03 2016

@author: Joanna
"""
import pandas as pd
import talib as ta

def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

def relative_strength(prices, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """

    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n - 1) + upval)/n
        down = (down*(n - 1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi

def moving_average_convergence(x, nslow=26, nfast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = moving_average(x, nslow, type='exponential')
    emafast = moving_average(x, nfast, type='exponential')
    return emaslow, emafast, emafast - emaslow

def bbands(price, length=30, numsd=2):
    """ returns average, upper band, and lower band"""
    ave = pd.stats.moments.rolling_mean(price,length)
    sd = pd.stats.moments.rolling_std(price,length)
    upband = ave + (sd*numsd)
    dnband = ave - (sd*numsd)
    return np.round(ave,3), np.round(upband,3), np.round(dnband,3)

def get_rsi_timeseries(prices, n=14):
    # RSI = 100 - (100 / (1 + RS))
    # where RS = (Wilder-smoothed n-period average of gains / Wilder-smoothed n-period average of -losses)
    # Note that losses above should be positive values
    # Wilder-smoothing = ((previous smoothed avg * (n-1)) + current value to average) / n
    # For the very first "previous smoothed avg" (aka the seed value), we start with a straight average.
    # Therefore, our first RSI value will be for the n+2nd period:
    #     0: first delta is nan
    #     1:
    #     ...
    #     n: lookback period for first Wilder smoothing seed value
    #     n+1: first RSI

    # First, calculate the gain or loss from one price to the next. The first value is nan so replace with 0.
    deltas = (prices-prices.shift(1)).fillna(0)

    # Calculate the straight average seed values.
    # The first delta is always zero, so we will use a slice of the first n deltas starting at 1,
    # and filter only deltas > 0 to get gains and deltas < 0 to get losses
    avg_of_gains = (deltas[1:n+1])[deltas[1:n+1] > 0].sum() / n
    avg_of_losses = -deltas[1:n+1][deltas[1:n+1] < 0].sum() / n

    # Set up pd.Series container for RSI values
    rsi_series = pd.Series(0.0, deltas.index)

    # Now calculate RSI using the Wilder smoothing method, starting with n+1 delta.
    up = lambda x: x if x > 0 else 0
    down = lambda x: -x if x < 0 else 0
    i = n+1
    for d in deltas[n+1:]:
        avg_of_gains = ((avg_of_gains * (n-1)) + up(d)) / n
        avg_of_losses = ((avg_of_losses * (n-1)) + down(d)) / n
        if avg_of_losses != 0:
            rs = avg_of_gains / avg_of_losses
            rsi_series[i] = 100 - (100 / (1 + rs))
        else:
            rsi_series[i] = 100
        i += 1

    return rsi_series
    
import numpy as np;
#import talib as ta;

def KDJ(high, low, close, kPeriods=14, dPeriods=3):
	vectorSize = high.shape[0];
	fastK = np.zeros(vectorSize);
	fastK[:] = np.NAN;
	lowestLowVector = np.zeros(vectorSize);
	lowestLowVector[0:kPeriods] = min(low[0:kPeriods]);
	for i in range((kPeriods-1), vectorSize):
		lowestLowVector[i] = min(low[(i-kPeriods+1):i+1]);
	highestHighVector = np.zeros(vectorSize);
	highestHighVector[0:kPeriods] = max(high[0:kPeriods]);
	for i in range(kPeriods-1, vectorSize):
		highestHighVector[i] = max(high[(i-kPeriods+1):i+1]);
	highLowDiff = highestHighVector - lowestLowVector;
	nonZero = highLowDiff.ravel().nonzero();
	fastK[nonZero] = np.divide((close[nonZero] - lowestLowVector[nonZero]),
							   (highestHighVector[nonZero]-lowestLowVector[nonZero])) * 100;
	fastD = np.zeros(vectorSize);
	fastD[:] = np.NAN;
	fastD[~np.isnan(fastK)] = moving_average(fastK[~np.isnan(fastK)], dPeriods, 'exponential');
	jLine = 3*fastK-2*fastD;
	
	return np.hstack((fastK.reshape((fastK.size, 1)), fastD.reshape((fastD.size, 1)),
					 jLine.reshape((jLine.size, 1))))
	
def TSI(close, slowPeriod=25, fastPeriod=13):
	if ((fastPeriod >= close.size) | (slowPeriod >= close.size)):
		return;
	momentumVector = np.zeros(close.size);
	momentumVector[1:] = close[1:] - close[0:(close.size-1)]
	absMomentumVector = np.abs(momentumVector);
	k1 = 2/(slowPeriod+1);
	k2 = 2/(fastPeriod+1);
	
	ema1 = np.zeros(close.size);
	ema2 = np.copy(ema1);
	ema3 = np.copy(ema1);
	ema4 = np.copy(ema1);
	
	for i in range(1,close.size):
		ema1[i] = k1 * (momentumVector[i]-ema1[i-1]) + ema1[i-1];
		ema2[i] = k2 * (ema1[i]-ema2[i-1])   + ema2[i-1];
		ema3[i] = k1 * (absMomentumVector[i]-ema3[i-1]) + ema3[i-1];
		ema4[i] = k2 * (ema3[i]-ema4[i-1])   + ema4[i-1];

	tsi = 100 * np.divide(ema2, ema4)
	return tsi;

def HHLL(high, low, periods=20):
	vectorSize = high.shape[0];
	lowestLowVector = np.zeros(vectorSize);
	lowestLowVector[0:periods] = min(low[0:periods]);
	for i in range((periods-1), vectorSize):
		lowestLowVector[i] = min(low[(i-periods+1):i+1]);
	highestHighVector = np.zeros(vectorSize);
	highestHighVector[0:periods] = max(high[0:periods]);
	for i in range((periods-1), vectorSize):
		highestHighVector[i] = max(high[(i-periods+1):i+1]);
	
	midPointVector = (highestHighVector + lowestLowVector) / 2;
	return np.hstack((highestHighVector.reshape((highestHighVector.size, 1)), lowestLowVector.reshape((lowestLowVector.size, 1)),
					 midPointVector.reshape((midPointVector.size, 1))))
	
def CMF(high, low, close, volume, periods=20):
	vectorSize = high.shape[0]; 
	moneyFlowMultiplier = ((close - low) - (high - close)).reshape((vectorSize,1)) * np.linalg.pinv((high - low).reshape((vectorSize,1)));
	moneyFlowVolume = np.dot(moneyFlowMultiplier,volume.reshape((vectorSize,1)));
	cmf = np.zeros(vectorSize);
	cmf[:] = np.NAN;
	for i in range((periods-1),vectorSize):
		cmf[i] = sum(moneyFlowVolume[i-periods+1:i+1]) / sum(volume[i-periods+1:i+1]);
	return cmf;

def FORCE(close, volume, periods=13):
	vectorSize = close.shape[0];
	force = np.zeros(vectorSize);
	force[0] = np.nan;
	force[1:] = (close[1:] - close[0:vectorSize-1]) * volume[1:];
	force[1:] = moving_average(force[1:], periods, 'exponential');
	return force;

def VR(high, low, close, periods=14):
	vectorSize = close.shape[0];
	highLowDiff = high - low;
	highCloseDiff = np.zeros(vectorSize);
	highCloseDiff[1:] = np.abs(np.add(high[1:],-close[0:vectorSize-1]));
	lowCloseDiff = np.zeros(vectorSize);
	lowCloseDiff[1:] = np.abs(np.add(low[1:],-close[0:vectorSize-1]));
	vectorsStacked = np.vstack((highLowDiff, highCloseDiff, lowCloseDiff));
	tr = np.amax(vectorsStacked, axis=0);
	vr = tr / moving_average(tr, periods,'exponential');
	return vr;
 
 #-------------test script-------------------------------------#
def get_TA_CdL_Func_List ():
    cdl_pattern_func_list = [ta.CDL2CROWS,
                            ta.CDL3BLACKCROWS,
                            ta.CDL3INSIDE,
                            ta.CDL3LINESTRIKE,
                            ta.CDL3OUTSIDE,
                            ta.CDL3STARSINSOUTH,
                            ta.CDL3WHITESOLDIERS,
                            ta.CDLABANDONEDBABY,
                            ta.CDLADVANCEBLOCK,
                            ta.CDLBELTHOLD,
                            ta.CDLBREAKAWAY,
                            ta.CDLCLOSINGMARUBOZU,
                            ta.CDLCONCEALBABYSWALL,
                            ta.CDLCOUNTERATTACK,
                            ta.CDLDARKCLOUDCOVER,
                            ta.CDLDOJI,
                            ta.CDLDOJISTAR,
                            ta.CDLDRAGONFLYDOJI,
                            ta.CDLENGULFING,
                            ta.CDLEVENINGDOJISTAR,
                            ta.CDLEVENINGSTAR,
                            ta.CDLGAPSIDESIDEWHITE,
                            ta.CDLGRAVESTONEDOJI,
                            ta.CDLHAMMER,
                            ta.CDLHANGINGMAN,
                            ta.CDLHARAMI,
                            ta.CDLHARAMICROSS,
                            ta.CDLHIGHWAVE,
                            ta.CDLHIKKAKE,
                            ta.CDLHIKKAKEMOD,
                            ta.CDLHOMINGPIGEON,
                            ta.CDLIDENTICAL3CROWS,
                            ta.CDLINNECK,
                            ta.CDLINVERTEDHAMMER,
                            ta.CDLKICKING,
                            ta.CDLKICKINGBYLENGTH,
                            ta.CDLLADDERBOTTOM,
                            ta.CDLLONGLEGGEDDOJI,
                            ta.CDLLONGLINE,
                            ta.CDLMARUBOZU,
                            ta.CDLMATCHINGLOW,
                            ta.CDLMATHOLD,
                            ta.CDLMORNINGDOJISTAR,
                            ta.CDLMORNINGSTAR,
                            ta.CDLONNECK,
                            ta.CDLPIERCING,
                            ta.CDLRICKSHAWMAN,
                            ta.CDLRISEFALL3METHODS,
                            ta.CDLSEPARATINGLINES,
                            ta.CDLSHOOTINGSTAR,
                            ta.CDLSHORTLINE,
                            ta.CDLSPINNINGTOP,
                            ta.CDLSTALLEDPATTERN,
                            ta.CDLSTICKSANDWICH,
                            ta.CDLTAKURI,
                            ta.CDLTASUKIGAP,
                            ta.CDLTHRUSTING,
                            ta.CDLTRISTAR,
                            ta.CDLUNIQUE3RIVER,
                            ta.CDLUPSIDEGAP2CROWS,
                            ta.CDLXSIDEGAP3METHODS]

    return cdl_pattern_func_list
     
     
     
     

 
