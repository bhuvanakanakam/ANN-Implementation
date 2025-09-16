layers_dims = [4,20,20,1]
mini_batch_size = 256
learning_rate = 0.001
num_iterations = 500
activation = "tanh"
cost_func = "mse"

optimizer = "adam" # none or momentum or adam
beta = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

regularisation = "L2" # none or L2
lambd = 0.1