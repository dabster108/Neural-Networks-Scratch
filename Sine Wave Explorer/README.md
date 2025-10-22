
# Sine Wave Explorer

## Overview
Sine Wave Explorer is a neural network project that combines **sine wave prediction** and **frequency recognition**. The model can:  
1. Predict the value of `sin(x)` for any input `x`.  
2. Identify the frequency of a sampled sine wave (1Hz, 2Hz, or 3Hz).  

This project is an excellent intermediate project to understand regression, classification, and signal patterns using neural networks.

---

## Features
- **Sine Predictor:** Learns the smooth sine curve using a small feedforward neural network.
- **Frequency Recognizer:** Classifies sampled sine wave data into 1Hz, 2Hz, or 3Hz.
- **Visualization:** Plots predicted vs real sine waves and shows frequency prediction results.

---

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- PyTorch or TensorFlow (depending on implementation)

---

## Project Structure

SineWaveExplorer/
│
├── neural.py      # Defines the neural network architecture
├── train.py       # Script to train the model for regression and classification
├── main.py        # Script to test the model and visualize predictions
└── README.md      # Project description and instructions

---

## How to Run
1. Train the model:

python train.py

2. Test and visualize:

python main.py

---

## Author
Dikshanta Chapagain


