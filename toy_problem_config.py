layers_dims = [1, 30, 30, 1]
mini_batch_size = 64
learning_rate = 0.001
num_iterations = 4000
activation = "tanh" # sigmoid or relu or tanh
cost_func = "mse"

regularisation = "L2" # none or L2
lambd = 0.1

optimizer = "momentum" # none or momentum or adam
beta = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8