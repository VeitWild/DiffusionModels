import jax.scipy.stats 
import jax.numpy as jnp
import numpy as np


#import jax.scipy 

class GMM:
    '''
    Implements a Gaussian Mixture Model 
    '''

    def __init__(self,mix_weight,means,covariances):
        self.mix_weight = mix_weight
        self.means = means #Dim-state-space x Nr_ components
        self.covariances = covariances # Dim-state-space x Dim state-pace x Nr_components
        self.dim_state_space = means.shape[0]

    def pdf(self,x):
        #x is  Nr_Evaluations x Dim-state-space
        w = self.mix_weight
        covs = self.covariances
        means = self.means

        density = 0

        for k in range(0,len(w)):
            density = density +  w[k]* jax.scipy.stats.multivariate_normal.pdf(x,mean = means[:,k], cov = covs[:,:,k] )

        return density

    def log_pdf(self,x):
        return jnp.log(self.pdf(x))
    
    def score_function(self,x):
        
        log_pdf = lambda x: jnp.log(self.pdf(x))

        return jax.jit(jax.vmap(jax.grad(log_pdf)))(x)

    
    def sample(self,N):
        nr_samples=np.random.multinomial(N, self.mix_weight)

        K=len(self.mix_weight)
        means = self.means
        covs = self.covariances

        sample_list = []

        for k in range(0,K):

            sample_list.append(np.random.multivariate_normal(means[:,k], covs[:,:,k], nr_samples[k]))
        
        samples = np.concatenate( sample_list, axis=0 )
        
        return samples


def initialise_means(K=4,lower=-3,upper=3):
    '''
    Sample K means from unifrom in [-lower,upper]**2
    '''
    return(np.random.uniform(low=lower,high=upper,size=(2,K)))

        
##Just some nonsense
#asd
