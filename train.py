# train.py
import numpy as np
from Neural_Netowrk_Bank_Dataset.data_clean import X_train, X_test, y_train, y_test
from Neural_Netowrk_Bank_Dataset.neural import initialize_network, forward, compute_loss, backward_pass, update_parameters

# Step 1: Initialize network
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1

W1, b1, W2, b2 = initialize_network(input_size, hidden_size, output_size)
params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# Step 2: Training loop
epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    # Forward pass
    Z1, A1, Z2, A2 = forward(X_train, params["W1"], params["b1"], params["W2"], params["b2"])

    # Compute loss
    loss = compute_loss(y_train.values, A2)

    # Backward pass
    cache = {"A1": A1, "A2": A2}
    grads = backward_pass(X_train, y_train.values, cache, params, Z1)

    # Update parameters
    params = update_parameters(params, grads, learning_rate)

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Step 3: Evaluation
def predict(X, params):
    _, _, _, A2 = forward(X, params["W1"], params["b1"], params["W2"], params["b2"])
    return (A2 >= 0.5).astype(int)

y_pred_train = predict(X_train, params)
y_pred_test = predict(X_test, params)

train_acc = np.mean(y_pred_train.flatten() == y_train.values)
test_acc = np.mean(y_pred_test.flatten() == y_test.values)

print(f"\nFinal Training Accuracy: {train_acc*100:.2f}%")
print(f"Final Test Accuracy: {test_acc*100:.2f}%")