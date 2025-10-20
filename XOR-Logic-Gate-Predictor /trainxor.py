# train.py
import numpy as np
from neural import network, forward, compute_loss, backward_pass, update_parameters, sigmoid, relu, relu_derivative

# XOR input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([0, 1, 1, 0])  # XOR output

# Initialize network
input_size = 2
hidden_size = 4
output_size = 1
W1, b1, W2, b2 = network(input_size, hidden_size, output_size)
params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# Training loop
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward pass
    Z1, A1, Z2, A2 = forward(X, params["W1"], params["b1"], params["W2"], params["b2"])
    cache = {"A1": A1, "A2": A2}

    # Compute loss
    loss = compute_loss(Y, A2)

    # Backward pass
    grads = backward_pass(X, Y, cache, params)

    # Update parameters
    params = update_parameters(params, grads, learning_rate)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


preds = (A2 >= 0.5).astype(int)
print("Final predictions on XOR inputs:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted: {preds[i][0]}, Actual: {Y[i]}")