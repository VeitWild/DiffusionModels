import sys
sys.path.append("/home/vdwild/Code/DiffusionModels")
from data import GMM, gmm
from flax.training import train_state, checkpoints

import jax.numpy as jnp
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from flax import linen as nn
from jax import jacfwd
import jax.random
import jax.numpy as jnp

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
        x = nn.sigmoid(x)
        x = self.hidden2(x)
        x = nn.sigmoid(x)
        x = self.out(x)
        return x

class GMMDataset(data.Dataset):

    def __init__(self, size, seed, gmm):
        """
        Inputs:
            size - Number of data points we want to generate
            seed - The seed to use to create the PRNG state with which we want to generate the data points
        """
        super().__init__()
        self.size = size
        self.np_rng = np.random.RandomState(seed=seed)
        self.gmm = gmm
        self.data = gmm.sample(self.size)
        
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        return data_point



dataset = GMMDataset(size=200, seed=42,gmm=gmm)
print("Size of dataset:", len(dataset))
print("Data point 0:", dataset[0])

##
def visualize_samples(data):

    plt.figure(figsize=(4,4))
    plt.scatter(data[:,0], data[:,1], edgecolor="#333")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

#visualize_samples(dataset.data)
#plt.show()


# This collate function is taken from the JAX tutorial with PyTorch Data Loading
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=numpy_collate)


# next(iter(...)) catches the first batch of the data loader
# If shuffle is True, this will return a different batch every time we run this cell
# For iterating over the whole dataset, we can simple use "for batch in data_loader: ..."
data_inputs = next(iter(data_loader))

# The shape of the outputs are [batch_size, d_1,...,d_N] where d_1,...,d_N are the
# dimensions of the data point returned from the dataset class
print("Data inputs", data_inputs.shape, "\n", data_inputs)

#Initialise Model
hidden = jnp.array([10,10])
model = score_function(num_hidden=hidden, num_outputs=2)
rng = jax.random.PRNGKey(42)
rng, inp_rng, init_rng = jax.random.split(rng, 3)
X = jax.random.normal(inp_rng, (8, 2))  # Batch size 8, input size 2
params = model.init(init_rng, X)

# Input to the optimizer are optimizer settings like learning rate
optimizer = optax.sgd(learning_rate=0.1)

model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)

def calculate_loss(state, params, batch):

    score_fn = lambda x: state.apply_fn(params,x)
    pred = score_fn(batch) #batch x dim_state_space (=2)
    jacobian =jax.vmap(jacfwd(score_fn))(batch) #batch x dim_state_space x dim_state_space

    loss = (jnp.trace(jacobian,axis1=1,axis2=2)).mean() + 0.5 * jnp.mean(jnp.square(pred).sum(axis=1))

    return loss

batch = next(iter(data_loader))
print(calculate_loss(model_state, model_state.params, batch))

#@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(calculate_loss,  # Function to calculate the loss
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=False  # Function has additional outputs, here not true
                                )
    # Determine gradients for current model, parameters and batch
    loss, grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss

@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    # Determine the accuracy
    kld = calculate_loss(state, state.params, batch)
    return kld

##Here comes the training
train_dataset = GMMDataset(size=250, seed=42,gmm=gmm)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=numpy_collate)

def train_model(state, data_loader, num_epochs=100):
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss = train_step(state, batch)
            # We could use the loss and accuracy for logging here, e.g. in TensorBoard
            # For simplicity, we skip this part here
    return state


trained_model_state = train_model(model_state, train_data_loader, num_epochs=100)

checkpoints.save_checkpoint(ckpt_dir='Code/DiffusionModels/models/my_checkpoints/',  # Folder to save checkpoint in
                            target=trained_model_state,  # What to save. To only save parameters, use model_state.params
                            step=100,  # Training step or other metric to save best model on
                            prefix='my_model',  # Checkpoint file name prefix
                            overwrite=True   # Overwrite existing checkpoint files
                           )

#print(params)

#Returns f_\theta(x) for theta = param

#print(model.apply(params, X))

#f = lambda x: model.apply(params,x)

#print(f(x))
#print(jacfwd(f)(X))
#jacobian =jax.vmap(jacfwd(f))(X)

#jacobian = jax.jit(jax.vmap(jacfwd(  model.apply,static_argnums=1  )  ) )(X)


#print(jacobian[0,:])


