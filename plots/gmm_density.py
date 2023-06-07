import sys
print(sys.path)
sys.path.append("/home/vdwild/Code/DiffusionModels")
print(sys.path)

import data

import numpy as np
import jax

import matplotlib.pyplot as plt

###Number of mixture components
Nr_components = 2
weights = 1/Nr_components * np.repeat(1 , Nr_components)

#Sample means unifromaly from hypercube [lower,upper]**2
#means = jnp.array([[3,3],[3, -3], [-3, -3], [-3,3 ]  ]).transpose()
means = data.initialise_means(K=Nr_components,lower = -10, upper=10)

#Same covariance for all covariances
sd = 1
cov_base = sd**2 * np.eye(2) #sd = 0.5
covs = np.array([cov_base for k in range(0,Nr_components)]).transpose()
#covs = np.repeat(cov_base,Nr_components)
#print(covs[:,:,0])

rng= jax.random.PRNGKey(42) #
rng, key = jax.random.split(rng, num=2)  # ensures that rerunning generates new random number

x = jax.random.normal(key,(100,2))

gmm = data.GMM(mix_weight=weights, means= means, covariances=covs)







### Plot samples
samples = gmm.sample(N=200)
#plt.plot(samples[:,0],samples[:,1],'o')
#plt.show()

### Plot Countour Lines of Density and samples
num = 100

mean_min = means.min(axis=1)
mean_max = means.max(axis=1)

x1_range = np.linspace(mean_min[0]-sd*3,mean_max[0]+sd*3,num=num)
x2_range = np.linspace(mean_min[1]-sd*3,mean_max[1]+sd*3,num=num)
x1, x2 = np.meshgrid(x1_range,x2_range)

X = np.concatenate( (x1.reshape((num**2,1)), x2.reshape((num**2,1))) ,axis=1) #Num**2 x 2
pdf = gmm.pdf(X)


fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 


Z = pdf.reshape(num,num)
cp = ax.contour(x1, x2, Z,levels=10)
ax.clabel(cp, inline=True, 
          fontsize=10)
ax.set_title('Contour Plot')
ax.set_xlabel('x1 ')
ax.set_ylabel('x2')

ax.plot(samples[:,0],samples[:,1],'o')

plt.show()

