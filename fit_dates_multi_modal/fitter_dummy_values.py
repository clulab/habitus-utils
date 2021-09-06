from numpy import random
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-colorblind')
#datefn from:https://stackoverflow.com/questions/39260616/generate-a-normal-distribution-of-dates-within-a-range
import time
import numpy


_DATE_RANGE = ('2017-01-01', '2018-01-01')
_DATE_FORMAT = '%Y-%m-%d'
_EMPIRICAL_SCALE_RATIO = 0.15
_DISTRIBUTION_SIZE = 1000
_SCALE_RATIO_FOR_DRAWING=1e8

GENERATE_DUMMY_DATA=True

numpy.random.seed(3)

time_range = tuple(time.mktime(time.strptime(d, _DATE_FORMAT))
                       for d in _DATE_RANGE)


#whenever you are given a date range, expand it,, include it in the big distribution
#whenever you see a single date, include it in the big distribution


distribution = None
if(GENERATE_DUMMY_DATA):
    distribution1 = numpy.random.normal(loc=(time_range[0] + time_range[1]/2) * 0.5,scale=(time_range[1] - time_range[0]) * _EMPIRICAL_SCALE_RATIO,size=_DISTRIBUTION_SIZE)
    distribution2 = numpy.random.normal(loc=(time_range[1]/2 + time_range[1]) * 0.5,scale=(time_range[1] - time_range[0]) * _EMPIRICAL_SCALE_RATIO,size=_DISTRIBUTION_SIZE)
    distribution = np.concatenate([distribution1, distribution2])
    distribution=np.true_divide(distribution, _SCALE_RATIO_FOR_DRAWING)

assert distribution is not None
sigma=3
mu=1193084540.5079513



fig, axes = plt.subplots(sharex='all',sharey='all',figsize=(13, 7))



count, bins, ignored = axes.hist(distribution, bins=30, density=True)

def reverse_bins(bins):
    dates=[]
    for t in numpy.sort(bins):
        indiv=reverse_dates(t)
        dates.append(indiv)
    return dates

#for printing axes purposes
def reverse_dates(date,pos=None):
    indiv=time.strftime(_DATE_FORMAT, time.localtime(date*_SCALE_RATIO_FOR_DRAWING))
    return indiv

#to plot the actual distribution for a given sigma and mu i.e when using dummy data
axes.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')

labels = axes.get_xticklabels()

plt.setp(labels, rotation=45, horizontalalignment='right')

title_custom="Fitting a kernel density estimation over a multimodal \ndistribution of dates in a given range\n (lines=bandwidth of kernel density function)"
axes.set(xlim=[bins[0], bins[len(bins)-1]], xlabel='Dates which are edges of each bin', ylabel='No of dates per bin ',
       title=title_custom)

axes.xaxis.set_major_formatter(reverse_dates)

all_bdw=(np.linspace(0.1,1,10)).round(2)
for bndw in all_bdw:
    kde=KernelDensity(kernel='gaussian',bandwidth=bndw).fit(X=distribution.reshape(-1, 1))
    log_density=kde.score_samples(bins.reshape(-1,1))
    axes.plot(bins,np.exp(log_density),label=bndw)
axes.legend()
plt.show()
