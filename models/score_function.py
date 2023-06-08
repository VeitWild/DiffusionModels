import jax.numpy as jnp
import flax
from flax import linen as nn
import jax.random

class score_function(nn.Module):
    num_hidden : jnp.array  # Number of hidden neurons
    num_outputs : int  # Number of output neurons

    def setup(self):
        # Create the modules we need to build the network
        # nn.Dense is a linear layer
        self.hidden1 = nn.Dense(features=self.num_hidden[0])
        self.hidden2 = nn.Dense(features=self.num_hidden[1])
        self.out = nn.Dense(features=self.num_outputs)

    def __call__(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.hidden1(x)
        x = nn.tanh(x)
        x = self.hidden2(x)
        x = nn.sigmoid(x)
        x = self.out(x)
        return x

hidden = jnp.array([10,10])


model = score_function(num_hidden=hidden, num_outputs=2)

#Need to initialse paramters now

#print(model(jnp.array([1,2,3])))
rng = jax.random.PRNGKey(42)
rng, inp_rng, init_rng = jax.random.split(rng, 3)
X = jax.random.normal(inp_rng, (8, 2))  # Batch size 8, input size 2
# Initialize the model
params = model.init(init_rng, X)
print(params)

#Returns f_\theta(x) for theta = param
print(model.apply(params, X))