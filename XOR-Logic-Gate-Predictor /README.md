🧠 XOR Logic Gate Predictor using Neural Network
📌 Project Overview
This project demonstrates how a simple neural network can learn the XOR (exclusive OR) logic gate — a fundamental problem that traditional linear models fail to solve.
The XOR problem is widely used in machine learning to show why non-linear activation functions and multi-layer networks are necessary for complex decision-making.
🧩 Problem Definition
Input A	Input B	XOR Output
0	0	0
0	1	1
1	0	1
1	1	0
The XOR gate outputs 1 only when exactly one of the inputs is 1, and 0 otherwise.
This logic cannot be learned using a simple linear classifier, because XOR is not linearly separable.
🎯 Objective
To build a neural network from scratch that:
Takes two binary inputs (0 or 1).
Learns the XOR pattern through training.
Predicts the correct output for all input combinations.
🧰 Technologies Used
Python
NumPy (for matrix operations)
Matplotlib (optional) – for visualizing training loss
No external datasets are used — the XOR truth table is manually defined.
⚙️ How It Works
Input Layer: Takes two binary inputs (A and B).
Hidden Layer: Introduces non-linearity using activation (e.g. sigmoid).
Output Layer: Produces a binary output representing XOR.
Loss Function: Measures prediction error.
Backpropagation: Adjusts weights to minimize the loss.
After training, the network learns to perfectly predict XOR outputs.
💡 Real-World Analogy
The XOR gate is more than just a logic function — it represents nonlinear relationships that exist in real-world problems like:
Detecting exclusive conditions (e.g., user clicks either "Yes" or "No" but not both).
Decision-making where two features interact nonlinearly.
Foundations of complex neural architectures that model real-life behaviors and signals.
🚀 Why This Project Matters
This project helps you understand:
Why deep learning is needed beyond linear regression or perceptrons.
How neural networks learn nonlinear boundaries.
The fundamentals of forward propagation and backpropagation.
🧩 Folder Structure
XOR_Neural_Network/
│
├── data_clean.py          # (optional) manually defines XOR data
├── neural.py              # defines neural network logic
├── train.py               # trains the model
├── main.py                # makes predictions
└── README.md              # project overview
🧠 Future Scope
Extend to other logic gates (AND, OR, NAND, NOR).
Visualize decision boundaries.
Build using TensorFlow or PyTorch for comparison.