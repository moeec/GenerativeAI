# Replace <MASK> with the appropriate code to complete the exercise.

import torch.nn as nn

class MyMLP(nn.Module):
    """My Multilayer Perceptron (MLP)

    Specifications:

        - Input layer: 784 neurons
        - Hidden layer: 128 neurons with ReLU activation
        - Output layer: 10 neurons with softmax activation

    """

    def __init__(self):
        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)   # Replace <MASK> with nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Replace <MASK> with dim=1

    def forward(self, x):
        # Pass the input to the second layer
        x = self.fc1(x)

        # Apply ReLU activation
        x = self.relu(x)

        # Pass the result to the final layer
        x = self.fc2(x)

        # Apply softmax activation
        x = self.softmax(x)
        
        return x

my_mlp = MyMLP()
print(my_mlp)
