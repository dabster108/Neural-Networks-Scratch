# main.py
import numpy as np
import pandas as pd
from Neural_Netowrk_Bank_Dataset.data_clean import scaler, df_encoded, X_train, X_test, y_train, y_test
from Neural_Netowrk_Bank_Dataset.neural import initialize_network, forward, compute_loss, backward_pass, update_parameters

# -------------------------
# Step 1: Train network
# ------------------------- 
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1

# Initialize parameters
W1, b1, W2, b2 = initialize_network(input_size, hidden_size, output_size)
params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    # Forward pass
    Z1, A1, Z2, A2 = forward(X_train, params["W1"], params["b1"], params["W2"], params["b2"])
    cache = {"Z1": Z1, "A1": A1, "A2": A2}

    # Loss
    loss = compute_loss(y_train.values, A2)

    # Backward pass
    grads = backward_pass(X_train, y_train.values, cache, params)

    # Update parameters
    params = update_parameters(params, grads, learning_rate)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# -------------------------
# Step 2: Prediction function
# -------------------------
def predict(X, params):
    _, _, _, A2 = forward(X, params["W1"], params["b1"], params["W2"], params["b2"])
    return (A2 >= 0.5).astype(int)


# -------------------------
# Step 3: Predict new data
# -------------------------
new_customer = {
    "age": 35,
    "job": "technician",
    "marital": "single",
    "education": "tertiary",
    "default": "no",
    "balance": 500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 12,
    "month": "may",
    "duration": 300,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

# Convert dict to DataFrame
new_df = pd.DataFrame([new_customer])

# Apply same encoding as training
new_encoded = pd.get_dummies(
    new_df,
    columns=['job','marital','education','default',
             'housing','loan','contact','month','poutcome'],
    drop_first=True
)

# Align with training columns
new_encoded = new_encoded.reindex(columns=df_encoded.drop('deposit', axis=1).columns, fill_value=0)

# Scale numeric features
new_scaled = scaler.transform(new_encoded)

# Make prediction
prediction = predict(new_scaled, params)

print("\nPrediction for new customer:",
      "Yes (will deposit)" if prediction[0][0] == 1 else "No (will not deposit)")