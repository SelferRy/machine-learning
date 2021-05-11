import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from sklearn.datasets import load_boston
from torch.utils.data import TensorDataset, DataLoader


# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    global train_dl
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)

            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('Training loss: ', loss_fn(model(inputs), targets))


if __name__ == "__main__":
    data = load_boston()

    inputs =  data.data
    targets = data.target

    inputs = torch.tensor(inputs, dtype=torch.float)  # torch.from_numpy(inputs).float()
    targets = torch.tensor(targets, dtype=torch.float).view(-1, 1)  # torch.from_numpy(targets).float()

    # Define dataset
    train_ds = TensorDataset(inputs, targets)

    # Define data loader
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    # Define model
    model = torch.nn.Linear(13, 1)  # inputs.shape[1

    # Define optimizer
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)

    # Define loss function
    loss_fn = F.mse_loss

    # Train the model for 100 epochs
    fit(100, model, loss_fn, opt)

    # Generate predictions
    preds = model(inputs)