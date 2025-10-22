# train.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from neural import create_model



# Regression: sin(x)
x_reg = np.linspace(-np.pi, np.pi, 1000).reshape(-1,1)
y_reg = np.sin(x_reg)

# Classification: frequency recognition
freqs = [1, 2, 3]
samples_per_freq = 200
x_class = []
y_class = []

for f in freqs:
    for _ in range(samples_per_freq):
        x_points = np.linspace(0, 2*np.pi, 20)
        y_points = np.sin(f * x_points)
        x_class.append(y_points)
        y_class.append(f-1)  # classes: 0,1,2

x_class = np.array(x_class, dtype=np.float32)
y_class = np.array(y_class, dtype=np.int64)

# Convert regression data to tensors
x_reg_tensor = torch.tensor(x_reg, dtype=torch.float32)
y_reg_tensor = torch.tensor(y_reg, dtype=torch.float32)

# Convert classification data to tensors
x_class_tensor = torch.tensor(x_class, dtype=torch.float32)
y_class_tensor = torch.tensor(y_class, dtype=torch.long)

# --------- Model and Loss ---------
model = create_model()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_regression = nn.MSELoss()
loss_classification = nn.CrossEntropyLoss()

# --------- Training Loop ---------
epochs = 500
for epoch in range(epochs):
    model.train()
    
    # Regression forward
    sin_pred, _ = model(x_reg_tensor, task='regression')
    loss_reg = loss_regression(sin_pred, y_reg_tensor)
    
    # Classification forward
    _, freq_pred = model(x_class_tensor, task='classification')
    loss_class = loss_classification(freq_pred, y_class_tensor)
    
    # Total loss
    loss = loss_reg + loss_class
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# --------- Save Model ---------
torch.save(model.state_dict(), "sinewave_model.pt")
print("Model saved as sinewave_model.pt")

# --------- Optional: Plot Learned Sine Curve ---------
model.eval()
with torch.no_grad():
    pred_sin, _ = model(x_reg_tensor)
    plt.plot(x_reg, y_reg, label="True sin(x)")
    plt.plot(x_reg, pred_sin.numpy(), label="Predicted sin(x)")
    plt.legend()
    plt.title("Sine Wave Prediction After Training")
    plt.show()