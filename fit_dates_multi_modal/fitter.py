from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('Solarize_Light2')
from scipy.stats.distributions import norm

y_axis = np.concatenate([norm(-1, 1.).rvs(400),norm(1, 0.3).rvs(100)])

#y_axis=np.array([2,4,3,7.5,4.5,7.8,2,4,3,7.5])
# The grid we'll use for plotting
x_grid = np.linspace(-2, 1, 500)

fig, axes = plt.subplots(sharex='all',sharey='all',)
#axes.set(xlim=(-10,10),ylim=(-10,10))
#x_axis=np.array(list(data.values()))
#y_axis=np.array(list(data.keys()))
axes.plot(x_grid,y_axis,'rx')
all_bdw=[np.linspace(0,1,100)]
for bndw in all_bdw:
    kde=KernelDensity(kernel='gaussian',bandwidth=bndw).fit(X=y_axis.reshape(-1,1))
    log_density=kde.score_samples(x_grid.reshape(-1,1))
    axes.plot(x_grid,(log_density),'b--')
plt.show()
