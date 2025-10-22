# main.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from Sine_Wave_Explorer.neural import create_model


model = create_model()
model.load_state_dict(torch.load("sinewave_model.pt"))
model.eval()

x_test = np.linspace(-np.pi, np.pi, 500).reshape(-1,1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

with torch.no_grad():
    sin_pred, _ = model(x_test_tensor)


plt.plot(x_test, np.sin(x_test), label="True sin(x)")
plt.plot(x_test, sin_pred.numpy(), label="Predicted sin(x)")
plt.legend()
plt.title("Sine Wave Prediction")
plt.show()

# --------- Classification: Frequency Recognition ---------
# Generate a test wave (choose 1Hz, 2Hz, or 3Hz)
freq = 2  # example
x_points = np.linspace(0, 2*np.pi, 20)
y_points = np.sin(freq * x_points).reshape(1,-1)
y_points_tensor = torch.tensor(y_points, dtype=torch.float32)

with torch.no_grad():
    _, freq_pred = model(y_points_tensor)

predicted_class = torch.argmax(freq_pred, dim=1).item() + 1
print(f"Actual Frequency: {freq} Hz, Predicted Frequency: {predicted_class} Hz")