import sys
sys.path.append("/home/vdwild/Code/DiffusionModels")


import numpy as np
import matplotlib.pyplot as plt
from data import initialise_means
from data import GMM

import jax

###Number of mixture components
Nr_components = 2   
weights = 1/Nr_components * np.repeat(1 , Nr_components)

#Sample means unifromaly from hypercube [lower,upper]**2


#means = jax.numpy.array([[-2,2],[2, -2] ]).transpose()
means = initialise_means(K=Nr_components,lower = -10, upper=10)

#Same covariance for all covariances
sd = 1
cov_base = sd**2 * np.eye(2) #sd = 0.5
#cov_base = [[1, 0.9],[0.9,1]]
covs = np.array([cov_base for k in range(0,Nr_components)]).transpose()



rng= jax.random.PRNGKey(42) #
rng, key = jax.random.split(rng, num=2)  # ensures that rerunning generates new random number

x = jax.random.normal(key,(100,2))

gmm = GMM(mix_weight=weights, means= means, covariances=covs)


### Plot Score function Lines of Density and samples
num = 10*Nr_components

mean_min = means.min(axis=1)
mean_max = means.max(axis=1)

x1_range = np.linspace(mean_min[0]-sd*3,mean_max[0]+sd*3,num=num)
x2_range = np.linspace(mean_min[1]-sd*3,mean_max[1]+sd*3,num=num)
x1, x2 = np.meshgrid(x1_range,x2_range)

X = np.concatenate( (x1.reshape((num**2,1)), x2.reshape((num**2,1))) ,axis=1) #Num**2 x 2
score_function = gmm.score_function(X)

plt.quiver(x1, x2, score_function[:,0], score_function[:,1], color='b')
plt.title('Score Function')
plt.show()