# neural.py
import numpy as np

# --------------------------
# Step 1: Initialize network
# --------------------------
def initialize_network(input_size, hidden_size=10, output_size=1):
    np.random.seed(42)  # reproducibility
    
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    
    return W1, b1, W2, b2

# --------------------------
# Step 2: Activation functions
# --------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# --------------------------
# Step 3: Forward pass
# --------------------------
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1  
    A1 = relu(Z1)             

    Z2 = np.dot(A1, W2) + b2  
    A2 = sigmoid(Z2)          

    return Z1, A1, Z2, A2

# --------------------------
# Step 4: Compute loss
# --------------------------
def compute_loss(Y, A2):
    m = Y.shape[0]
    loss = -np.mean(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))
    return loss

# --------------------------
# Step 5: Backward pass
# --------------------------
def backward_pass(X, Y, cache, params, Z1):
    m = X.shape[0]

    A1 = cache["A1"]
    A2 = cache["A2"]

    # Output layer
    dZ2 = A2 - Y.reshape(-1, 1)   
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer (using ReLU derivative now ✅)
    dZ1 = np.dot(dZ2, params["W2"].T) * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

# --------------------------
# Step 6: Update parameters
# --------------------------
def update_parameters(params, grads, learning_rate=0.01):
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params