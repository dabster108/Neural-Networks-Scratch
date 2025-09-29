# neural.py
import numpy as np

# --------------------------
# Step 1: Initialize network
# --------------------------
def initialize_network(input_size, hidden_size=10, output_size=1):
    """
    Initialize weights and biases for a 2-layer neural network.
    """
    np.random.seed(42)  # for reproducibility
    
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    
    return W1, b1, W2, b2


# --------------------------
# Step 2: Activation functions
# --------------------------

def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid activation: squashes values between 0 and 1"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

'''
ReLU (for hidden layer): helps with non-linearity and avoids vanishing gradients.
	•	Sigmoid (for output): maps values between 0 and 1, perfect for binary classification (like deposit: yes/no).'''



# --------------------------
# Step 3: Forward pass
# --------------------------
def forward(X, W1, b1, W2, b2):
    """
    Perform forward propagation through the network.
    X: input data (m x input_size)
    Returns: Z1, A1, Z2, A2
    """
    # Hidden layer
    Z1 = np.dot(X, W1) + b1   # linear step
    A1 = relu(Z1)             # activation

    # Output layer
    Z2 = np.dot(A1, W2) + b2  # linear step
    A2 = sigmoid(Z2)          # activation (probabilities)

    return Z1, A1, Z2, A2