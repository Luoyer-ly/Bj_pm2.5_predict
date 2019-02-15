import numpy as np
import matplotlib.pyplot as plt

def initialize_coef_deep(layers):
    layer_size = len(layers)
    parameters = {}
    for i in range(1, layer_size):
        parameters["W" + str(i)] = np.ones((layers[i], layers[i - 1]), dtype=float)
        parameters["b" + str(i)] = np.zeros((layers[i], 1))
    return parameters


def forward_propagation(X, parameters):
    layer_size = len(parameters) // 2
    A = X
    caches = {}
    caches["A0"] = X
    for i in range(1, layer_size):
        A, Z = linear_activation_forward(A, parameters["W" + str(i)], parameters["b" + str(i)], "relu")
        caches["A" + str(i)] = A
        caches["Z" + str(i)] = Z
    AL, ZL = linear_activation_forward(A, parameters["W" + str(layer_size)], parameters["b" + str(layer_size)],
                                       "tanh")
    caches["A" + str(layer_size)] = AL
    caches["Z" + str(layer_size)] = ZL
    return AL, caches


def L_layer_model(X, Y, layer_dims, iteration=2000, learning_rate=0.009, print_cost=True):
    m = Y.shape[1]
    parameters = initialize_coef_deep(layer_dims)
    for i in range(iteration):
        AL, caches = forward_propagation(X, parameters)
        if i % 100 == 0 and print_cost:
            cost = compute_cost(m, AL, Y)
            print("Cost after %i iterations: %f" % (i, cost))
        grads = backward_propagation(parameters, AL, caches, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
    return parameters


def compute_cost(m, AL, Y):
    cost = np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))
    cost = -np.sum(cost, axis=1, keepdims=True) / m
    return cost


def backward_propagation(parameters, AL, caches, Y):
    Y = Y.reshape(AL.shape)
    m = Y.shape[1]
    dAL = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
    # dAL = np.sum(dAL, axis=1, keepdims=True) / m
    L = len(parameters) // 2
    grads = {}
    dA_previous, dW, db = linear_activation_backward(dAL, caches["A" + str(L - 1)], caches["Z" + str(L)],
                                                     parameters["W" + str(L)],
                                                     parameters["b" + str(L)], "tanh")
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    for i in reversed(range(1, L)):
        dA_previous, dW, db = linear_activation_backward(dA_previous, caches["A" + str(i - 1)], caches["Z" + str(i)],
                                                         parameters["W" + str(i)],
                                                         parameters["b" + str(i)], "relu")
        grads["dW" + str(i)] = dW
        grads["db" + str(i)] = db
    return grads


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    return Z


def linear_activation_forward(A_prev, W, b, activation):
    Z = linear_forward(A_prev, W, b)
    if activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'relu':
        A = relu(Z)
    elif activation == 'tanh':
        A = tanh(Z)
    return A, Z


def linear_activation_backward(dA, A_previous, Z, W, b, activation):
    # A_prev = linear_backward(A, W, b)
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    elif activation == "relu":
        dZ = relu_backward(dA, Z)
    elif activation == 'tanh':
        dZ = tanh_backward(dA, Z)
    dA_previous, dW, db = linear_backward(dZ, A_previous, W, b)
    return dA_previous, dW, db


def update_parameters(parameters, grads, learning_rate):
    m = len(parameters) // 2
    for i in range(1, m + 1):
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * grads["dW" + str(i)]
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * grads["db" + str(i)]
    return parameters


def linear_backward(dZ, A_previous, W, b):
    m = dZ.shape[1]
    dW = 1 / m * np.dot(dZ, A_previous.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_previous = np.dot(W.T, dZ)
    return dA_previous, dW, db


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def sigmoid_backward(dA, Z):
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)
    return dZ


def relu(A):
    return np.maximum(0, A)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def tanh(A):
    return np.divide(np.exp(A) - np.exp(-A), np.exp(A) + np.exp(-A))


def tanh_backward(dA, Z):
    return dA * (1 - np.power(tanh(Z), 2))
