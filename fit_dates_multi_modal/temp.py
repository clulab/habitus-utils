from numpy import random
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('Solarize_Light2')

import time
import numpy

_DATE_RANGE = ('2000-05-12', '2020-05-12')
_DATE_FORMAT = '%Y-%m-%d'
_EMPIRICAL_SCALE_RATIO = 0.15
_DISTRIBUTION_SIZE = 1000


time_range = tuple(time.mktime(time.strptime(d, _DATE_FORMAT))
                       for d in _DATE_RANGE)

distribution = numpy.random.normal(
    loc=(time_range[0] + time_range[1]) * 0.5,
    scale=(time_range[1] - time_range[0]) * _EMPIRICAL_SCALE_RATIO,
    size=_DISTRIBUTION_SIZE
)
#s = np.array(list(time.strftime(_DATE_FORMAT, time.localtime(t))for t in numpy.sort(distribution)))

distribution=np.true_divide(distribution, 1e8)
sigma=0.1
mu=1193084540.5079513
#s=random.normal(mu,sigma, size=10000)


fig, axes=plt.subplots()
count, bins, ignored = axes.hist(distribution, 30, density=True)
axes.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')

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