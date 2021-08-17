from numpy import random
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('Solarize_Light2')
#datefn from:https://stackoverflow.com/questions/39260616/generate-a-normal-distribution-of-dates-within-a-range
import time
import numpy

_DATE_RANGE = ('2000-05-12', '2020-05-12')
_DATE_FORMAT = '%Y-%m-%d'
_EMPIRICAL_SCALE_RATIO = 0.15
_DISTRIBUTION_SIZE = 1000
_SCALE_RATIO_FOR_DRAWING=1e8

time_range = tuple(time.mktime(time.strptime(d, _DATE_FORMAT))
                       for d in _DATE_RANGE)

distribution = numpy.random.normal(
    loc=(time_range[0] + time_range[1]) * 0.5,
    scale=(time_range[1] - time_range[0]) * _EMPIRICAL_SCALE_RATIO,
    size=_DISTRIBUTION_SIZE
)


#s = np.array(list(time.strftime(_DATE_FORMAT, time.localtime(t))for t in numpy.sort(distribution)))

distribution=np.true_divide(distribution, _SCALE_RATIO_FOR_DRAWING)
sigma=3
mu=1193084540.5079513



fig, axes = plt.subplots(sharex='all',sharey='all')



count, bins, ignored = axes.hist(distribution, bins=30, density=True)

def reverse_bins(bins):
    dates=[]
    for t in numpy.sort(bins):
        indiv=reverse_dates(t)
        dates.append(indiv)
    return dates


def reverse_dates(date,pos=None):
    indiv=time.strftime(_DATE_FORMAT, time.localtime(date*_SCALE_RATIO_FOR_DRAWING))
    return indiv




print(reverse_bins(bins))



axes.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')

labels = axes.get_xticklabels()
axes.xaxis.set_major_formatter(reverse_dates)
plt.setp(labels, rotation=45, horizontalalignment='right')
#axes.plot(bins,distribution,'bx')
all_bdw=np.linspace(0.1,1,10)
#for bndw in [0.1,0.2,0.9]:
for bndw in all_bdw:
    kde=KernelDensity(kernel='gaussian',bandwidth=bndw).fit(X=distribution.reshape(-1, 1))
    log_density=kde.score_samples(bins.reshape(-1,1))
    axes.plot(bins,np.exp(log_density),'b--')
plt.show()
# figure,axes= plt.subplots()
# axes.set(xlim=[3,-3],ylim=[3,-3])
# axes.plot(y,'r--')
# plt.show()