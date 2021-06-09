import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# Download training dataset in tensor's form
dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

# Load an additonal test set of 10,000 images
test_dataset = MNIST(root='data/', train=False)

# Split dataset of train and validation sets
train_ds, val_ds = random_split(dataset, [50000, 10000])

# Create batches
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

# Model
input_size = 28 * 28
num_classes = 10


# class MnistModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(input_size, num_classes)
#
#     def forward(self, xb):
#         xb = xb.reshape(-1, 784)
#         out = self.linear(xb)
#         return out
#
#
# model = MnistModel()


# for images, labels in train_loader:
#     outputs = model(images)
#     break
#
# # Add activation function
# probs = F.softmax(outputs, dim=1)
# max_probs, preds = torch.max(probs, dim=1)

# Add Evalution Metric
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Add Loss function
loss_fn = F.cross_entropy

# Loss for current betch of data
loss = loss_fn(outputs, labels)


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine loss
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch,
                                                                     result['val_loss'],
                                                                     result['val_acc']))


model = MnistModel()


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, bal_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):

        # train
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # validation
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


