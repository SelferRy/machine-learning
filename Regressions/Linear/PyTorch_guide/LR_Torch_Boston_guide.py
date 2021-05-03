from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch


# Load data
boston = load_boston()
df = pd.DataFrame(boston.data)
df['Price'] = boston.target

# Normalized
data = df[df.columns[:-1]]
deta = data.apply(lambda x: (x - x.mean()) / x.std())
data["Price"] = df.Price

# Define features and labels
X = data.drop('Price', axis=1).to_numpy()
Y = data['Price'].to_numpy()

# Split dataset on train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Get torch-tensors
m = X_train.shape[0]
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
Y_train = torch.tensor(Y_train, dtype=torch.float).view(-1, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float).view(-1, 1)

# Define Neural Network
n = X_train.shape[1]
net = torch.nn.Sequential(torch.nn.Linear(n, 1))

# Weights and bias init
torch.nn.init.normal_(net[0].weight, mean=0, std=0.1)
torch.nn.init.constant_(net[0].bias, val=0)

# Union X_train and Y_train to datasets
datasets = torch.utils.data.TensorDataset(X_train, Y_train)
train_iter = torch.utils.data.DataLoader(datasets, batch_size=10, shuffle=True)

loss = torch.nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

num_epochs = 5
for epoch in range(num_epochs):
    for x, y in train_iter:
        y_pred = net(x)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print("epoch {} loss: {:.4f}".format(epoch + 1, l.item()))


