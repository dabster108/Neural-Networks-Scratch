import numpy as np 
def network(input_size, hidden_size=2, output_size=1):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

''' Layers
	1.	Input layer → receives the data (like numbers, images, or words).
	2.	Hidden layer(s) → processes the data and finds patterns.
	3.	Output layer → produces the result or prediction. ''' 

# Using the activation function sigmoid - results the value in probability between 0 and 1 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)  # derivative of sigmoid for backpropagation

def relu(x):
    return np.maximum(0, x)  # ReLU activation for hidden layer

def relu_derivative(x):
    return (x > 0).astype(float)  # derivative of ReLU for backpropagation

'''
sigmoid_derivative: tells the network how much to change weights at output layer.
relu: activates hidden neurons only if input > 0.
relu_derivative: tells the network how much to change weights at hidden layer.
'''

def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)  # hidden layer activation
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)  # output layer activation
    return Z1, A1, Z2, A2

def compute_loss(Y, A2):
    m = Y.shape[0]
    loss = -np.mean(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))
    return loss  # compute binary cross-entropy loss

def backward_pass(X, Y, cache, params):
    m = X.shape[0]
    A1, A2 = cache["A1"], cache["A2"]
    dZ2 = A2 - Y.reshape(-1, 1)  # error at output
    dW2 = (1/m) * np.dot(A1.T, dZ2)  # gradient for W2
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)  # gradient for b2
    dZ1 = np.dot(dZ2, params["W2"].T) * relu_derivative(A1)  # error for hidden
    dW1 = (1/m) * np.dot(X.T, dZ1)  # gradient for W1
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)  # gradient for b1
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads  # return all gradients

def update_parameters(params, grads, learning_rate=0.1):
    params["W1"] -= learning_rate * grads["dW1"]  # update W1
    params["b1"] -= learning_rate * grads["db1"]  # update b1
    params["W2"] -= learning_rate * grads["dW2"]  # update W2
    params["b2"] -= learning_rate * grads["db2"]  # update b2
    return params  # return updated parameters