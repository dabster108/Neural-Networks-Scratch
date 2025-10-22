
import torch
import torch.nn as nn
import torch.nn.functional as F


class SineWaveNet(nn.Module):
    def __init__(self):
        super(SineWaveNet, self).__init__()
      
        # For regression (single input)
        self.hidden1_reg = nn.Linear(1, 32)
        
        # For classification (20 inputs - sampled sine wave points)
        self.hidden1_class = nn.Linear(20, 32)
        
        # Shared layers
        self.hidden2 = nn.Linear(32, 64)      
        self.hidden3 = nn.Linear(64, 32)
        
        
        self.output_regression = nn.Linear(32, 1) 
        self.output_classification = nn.Linear(32, 3)  
    
    
    def forward(self, x, task='regression'):
        # Choose the appropriate first layer based on task
        if task == 'regression':
            x = F.relu(self.hidden1_reg(x))
        else:  # classification
            x = F.relu(self.hidden1_class(x))
        
        # Pass through shared hidden layers
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        
        # Two outputs
        sin_pred = self.output_regression(x)
        freq_pred = self.output_classification(x)
        
        return sin_pred, freq_pred

# Example function to create the model
def create_model():
    """
    Returns an instance of SineWaveNet.
    """
    model = SineWaveNet()
    return model