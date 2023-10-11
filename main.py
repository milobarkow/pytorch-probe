import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()  # Call the constructor of the parent class
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function  
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def train(self, input_data, target_data):
        # Define a loss function (Binary Cross Entropy) and an optimizer (Stochastic Gradient Descent)
        criterion = nn.BCELoss()  # Binary Cross Entropy loss function
        optimizer = optim.SGD(self.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer
        for epoch in range(10000):
            # Forward pass
            outputs = self(input_data)
            loss = criterion(outputs, target_data)  # Calculate the loss

            # Backpropagation and optimization
            optimizer.zero_grad()  # Zero the gradients to prevent accumulation
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update model weights using the optimizer

            if (epoch + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/10000], Loss: {loss.item():.4f}')

    def test(self, test_data):
        with torch.no_grad():
            predictions = self(test_data)
            print(predictions)


class CustomNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomNN, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train(self, input_data, target_data, show_loss=False):
        # Define a loss function (Binary Cross Entropy) and an optimizer (Stochastic Gradient Descent)
        criterion = nn.BCELoss()  # Binary Cross Entropy loss function
        optimizer = optim.SGD(self.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer
        for epoch in range(10000):
            # Forward pass
            outputs = self(input_data)
            loss = criterion(outputs, target_data)  # Calculate the loss

            # Backpropagation and optimization
            optimizer.zero_grad()  # Zero the gradients to prevent accumulation
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update model weights using the optimizer

            if show_loss and (epoch + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/10000], Loss: {loss.item():.4f}')

    def test(self, test_data):
        with torch.no_grad():
            predictions = self(test_data)
            print(predictions)


if __name__ == '__main__':
    input_size = 2
    hidden_size = [4, 5, 5, 4]
    output_size = 1

    # model = SimpleNN(input_size, hidden_size, output_size)
    model = CustomNN(input_size, hidden_size, output_size)

    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # Input data
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)  # Target data

    model.train(x, y)

    model.test(x)
