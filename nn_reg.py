import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
from nn_reg_configuration import *

np.random.seed(1)

lower = -2*math.pi
upper = 2*math.pi
interval = upper - lower
count = 1000
step = interval/count

def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims) - 1
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
    return parameters


def linear_forward(A, W, b):
    Z = W.dot(A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters,activation):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation)
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))      
    return AL, caches

def compute_cost(AL,Y,parameters,lambd,regularisation,cost_func='mse'):
    m = Y.shape[1]
    L = len(parameters) // 2     
    epsilon=0.001
    if (cost_func=='log'):
        cost = (1./m) * np.sum(-np.dot(Y,np.log(AL+epsilon).T) - np.dot(1-Y, np.log(1-AL+epsilon).T))
    if (cost_func=='mape'):
        cost=np.mean(np.abs((Y-AL)/(Y+epsilon)))*100
    if (cost_func=='mse'):
        cost=np.mean(np.square(AL-Y))*0.5
    if(regularisation=='L2'):
        sumw=0
        for l in range(1, L + 1):
            sumw=sumw+np.sum(np.square(parameters['W' + str(l)]))
        L2_regularization_cost = (1/m)*(lambd/2)*(sumw)  
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    if(regularisation=='L2'):
        cost=cost+L2_regularization_cost
    return cost

def mape_cost(Y,AL):
    epsilon=0.001
    cost=np.mean(np.abs((Y-AL)/(Y+epsilon)))*100
    return cost

def linear_backward(dZ, cache, regularisation, lambd):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = 1./m * np.dot(dZ, A_prev.T)
    
    if regularisation == 'L2':
        dW += lambd * W * (1/m)
    
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, regularisation, lambd, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache, regularisation, lambd)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, activation, regularisation, lambd, cost_func='log'):
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    if cost_func == 'log':
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    elif cost_func == 'mse':
        dAL = (AL - Y)
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, regularisation, lambd, activation)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, regularisation, lambd, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

def predict_vals(x, parameters,activation):
    probas, caches = L_model_forward(x, parameters,activation)
    return probas

def predicterr(x,y,parameters,lambd,activation,regularisation,cost_func):
    probas, caches = L_model_forward(x, parameters,activation)
    err=compute_cost(probas,y,parameters=parameters,lambd=lambd,regularisation=regularisation,cost_func=cost_func)
    return err
 
def adam_initalizer(parameters) :
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
    return v, s

def adam_parameters_update(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    L = len(parameters) // 2                 
    v_corrected = {}                         
    s_corrected = {}                         
    for l in range(L):
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] +(1-beta1)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1*v["db" + str(l+1)] +(1-beta1)*grads["db" + str(l+1)]
        v_corrected["dW" + str(l+1)] =  v["dW" + str(l+1)]/(1-math.pow(beta1,t))
        v_corrected["db" + str(l+1)] =  v["db" + str(l+1)]/(1-math.pow(beta1,t))
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] +(1-beta2)*grads["dW" + str(l+1)]*grads["dW" + str(l+1)]
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)] +(1-beta2)*grads["db" + str(l+1)]*grads["db" + str(l+1)]
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-math.pow(beta2,t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-math.pow(beta2,t))
        parameters["W" + str(l+1)] =  parameters["W" + str(l+1)]- learning_rate*v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)]+epsilon))
        parameters["b" + str(l+1)] =  parameters["b" + str(l+1)]- learning_rate*v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)]+epsilon))
    return parameters, v, s

def velocity_initalizer(parameters):
    L = len(parameters) // 2 
    v = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
    return v

def momemtum_parameters_update(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] +(1-beta)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = v["db" + str(l+1)]*beta +(1-beta)*grads["db" + str(l+1)]
        parameters["W" + str(l+1)] =  parameters["W" + str(l+1)]-learning_rate*(v["dW" + str(l+1)])
        parameters["b" + str(l+1)] =  parameters["b" + str(l+1)]-learning_rate*(v["db" + str(l+1)])
    return parameters, v

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def model_minibatches(X, Y,layers_dims,valid=False,valid_x=None,valid_y=None, optimizer='none', learning_rate = 0.0007,he_init=False, 
                        mini_batch_size = 64, beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_iterations = 10000,
                        activation='sigmoid',regularisation='none',print_cost = True,lambd=0.1, cost_func='mse'):
    L = len(layers_dims)
    costs = []
    validcosts=[]
    t = 0
    seed = 10
    m = X.shape[1]
    batches=m//mini_batch_size
    parameters=initialize_parameters(layers_dims)
    
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = velocity_initalizer(parameters)
    elif optimizer == "adam":
        v, s = adam_initalizer(parameters)
   
    for i in range(num_iterations):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a3, caches =  L_model_forward(X=minibatch_X,parameters= parameters,activation= activation)
            cost_total += compute_cost(AL= a3, Y= minibatch_Y,parameters= parameters,lambd= lambd,regularisation= regularisation,cost_func= cost_func)
            grads =  L_model_backward(AL= a3,Y= minibatch_Y,caches= caches,activation= activation,regularisation= regularisation,lambd=lambd,cost_func= cost_func)
            if optimizer == "gd" or optimizer=='none':
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = momemtum_parameters_update(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = adam_parameters_update(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)
        cost_avg = cost_total / batches
        
        if print_cost and i % 1 == 0:
            if(valid==True):
                valid_err=predicterr(valid_x,valid_y,parameters=parameters,lambd=lambd,activation=activation,regularisation='none',cost_func=cost_func)
                validcosts.append(valid_err)
                print ("Cost after epoch %i: %f,  Validation error: %f" %(i, cost_avg,valid_err))
            else:
                print ("Cost after epoch %i: %f" %(i, cost_avg))
            costs.append(cost_avg)
                
    # plot the cost
    plt.plot(costs)
    if(valid==True):
        plt.plot(validcosts)
        plt.legend(["train", "validation"], loc ="upper right") 
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title("Convergence history")
    plt.show()

    return parameters


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ

def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def tanh(Z):
    A = np.tanh(Z)
    cache = Z
    return A, cache

def tanh_backward(dA, cache):
    Z = cache
    s = np.tanh(Z)
    dZ = dA * (1 - s * s)
    assert (dZ.shape == Z.shape)
    return dZ

df = pd.read_csv("nn.csv")
df=((df-df.min())/(df.max()-df.min())) * (0.9 - (-0.9)) + (-0.9)
df = df.dropna().reset_index(drop=True)
test_size = round(df.shape[0]/10)
test = df[-test_size:]
rem_size = (df.shape[0] - test.shape[0])
rem = df[:rem_size]

train = rem[rem.index % 5 != 0] 
val = rem[rem.index % 5 == 0] 

x_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
x_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values
x_val = val.iloc[:, :-1].values
y_val = val.iloc[:, -1].values

x_train = np.transpose(x_train)
y_train = np.transpose(y_train)
y_train = y_train.reshape(1, -1)
x_test = np.transpose(x_test)
y_test = np.transpose(y_test)
y_test = y_test.reshape(1, -1)
x_val = np.transpose(x_val)
y_val = np.transpose(y_val)
y_val = y_val.reshape(1, -1)

print("No. of features: ", x_train.shape[0])
print("No. of samples in training set: ", x_train.shape[1])
print("No. of samples in testing set: ", x_test.shape[1])
print("No. of samples in validation set: ", x_val.shape[1])

parameters = model_minibatches(x_train, y_train, layers_dims,
                valid = True, valid_x = x_val, valid_y = y_val, num_iterations = num_iterations,
                he_init = True, mini_batch_size = mini_batch_size, learning_rate = learning_rate, print_cost = True,
                regularisation = regularisation, lambd = lambd, 
                optimizer = optimizer, beta = beta, beta1 = beta1, beta2 = beta2, epsilon = epsilon,
                activation = activation, cost_func = cost_func)

pred_val = predict_vals(x_val, parameters, activation=activation)
mape_val = mape_cost(y_val, pred_val)
print("MAPE - Validation : ", mape_val)

pred_test = predict_vals(x_test, parameters, activation=activation)
mape_test = mape_cost(y_test, pred_test)
print("MAPE - Test : ", mape_test)

preds, cache = L_model_forward(x_test, parameters, "tanh")
df2 = pd.DataFrame()
df2['predicted'] = preds[0].T.tolist()
df2['actual'] = y_test[0].T.tolist()

r2 = r2_score(y_test.T, preds.T)
print('the r2 score generated is: ', r2)

plt.scatter(y_test.T,preds.T)
plt.xlabel("True values")
plt.ylabel("Predicted values")
txt = "R^2 value = " + str(r2)
plt.text(-0.9, 0.6, txt, fontsize = 10)
plt.show()

data = np.hstack((y_test.T, preds.T))
result = pd.DataFrame(data=data,columns=['true','pred'])
result.to_csv("result.csv")