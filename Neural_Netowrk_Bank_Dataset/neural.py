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
    Z1 = np.dot(X, W1) + b1  
    A1 = relu(Z1)             

    # Output layer
    Z2 = np.dot(A1, W2) + b2  
    A2 = sigmoid(Z2)          

    return Z1, A1, Z2, A2



# --------------------------
# Step 4: Compute loss
# --------------------------
def compute_loss(Y, A2):
    """
    Compute Binary Cross-Entropy loss.
    Y: true labels (m, )
    A2: predicted probabilities (m, 1)
    """
    m = Y.shape[0]  # number of samples
    # Add a small epsilon (1e-8) to avoid log(0)
    loss = -np.mean(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))
    return loss




# --------------------------
# Step 5: Backward pass
# --------------------------
def backward_pass(X, Y, cache, params):
    """
    Perform backward propagation.
    X: input features (m, n_x)
    Y: true labels (m,)
    cache: values from forward pass
    params: current weights & biases
    """
    m = X.shape[0]  # number of samples

    # Extract cached values
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Gradients for output layer
    dZ2 = A2 - Y.reshape(-1, 1)   # (m, 1)
    dW2 = (1/m) * np.dot(A1.T, dZ2)  # (n_h, 1)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)  # (1, 1)

    # Gradients for hidden layer
    dZ1 = np.dot(dZ2, params["W2"].T) * (1 - A1**2)  # (m, n_h)
    dW1 = (1/m) * np.dot(X.T, dZ1)  # (n_x, n_h)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)  # (1, n_h)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads



# --------------------------
# Step 6: Update parameters
# --------------------------
def update_parameters(params, grads, learning_rate=0.01):
    """
    Update weights and biases using gradient descent.
    """
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    
    return params




'''
Step 1: Import the libraries and files we need.
We bring in NumPy for calculations, the cleaned dataset, and the neural network functions.

Step 2: Set the network size and initialize weights and biases.
We decide how many inputs, hidden neurons, and output neurons we want. Then we give small random numbers to weights and zeros to biases.

Step 3: Start the training loop.
We tell the network how many times to look at the data and learn from it.

Step 4: Do the forward pass.
The network calculates the hidden layer outputs and the final prediction probabilities.

Step 5: Calculate the loss.
We measure how different the predictions are from the real answers using a loss function.

Step 6: Do the backward pass.
The network finds the direction to change the weights to reduce the error.

Step 7: Update the weights and biases.
We adjust the weights and biases a little bit using the learning rate.

Step 8: Repeat steps 4–7 for many epochs.
The network keeps learning and the loss usually gets smaller.

'''