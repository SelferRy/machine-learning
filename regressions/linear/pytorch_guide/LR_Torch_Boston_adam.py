import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load data
boston = load_boston()
df = pd.DataFrame(boston.data)
df['Price'] = boston.target

# Normalized
data = df[df.columns[:-1]]
data = data.apply(lambda x: (x - x.mean()) / x.std())
data["Price"] = df.Price

# Define features and labels
X = data.drop('Price', axis=1).to_numpy()
Y = data['Price'].to_numpy()

# Split dataset on train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Get torch-tensors
m = X_train.shape[0]
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.FloatTensor(Y_train).view(-1, 1)
Y_test = torch.FloatTensor(Y_test).view(-1, 1)

# And make a convenient variable to remember the number of input columns
n = X.shape[0]


### Model definition ###

# First we define the trainable parameters A and b
A = torch.randn((1, n), requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Then we define the prediction model
def model(x_input):
    return A.mm(x_input) + b


### Loss function definition ###

def loss(y_predicted, y_target):
    return ((y_predicted - y_target)**2).sum()

### Training the model ###

# Setup the optimizer object, so it optimizes a and b.
optimizer = optim.Adam([A, b], lr=0.1)

# Main optimization loop
for t in range(2000):
    # Set the gradients to 0.
    optimizer.zero_grad()
    # Compute the current predicted y's from x_dataset
    y_predicted = model(x_dataset)
    # See how far off the prediction is
    current_loss = loss(y_predicted, y_dataset)
    # Compute the gradient of the loss with respect to A and b.
    current_loss.backward()
    # Update A and b accordingly.
    optimizer.step()
    print(f"t = {t}, loss = {current_loss}, A = {A.detach().numpy()}, b = {b.item()}")