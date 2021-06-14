import numpy as np
import torch.nn.functional as F


class EmptyModel:

    def fit(self, features, labels, num_epochs, opt):
        for epoch in range(num_epochs):
            pred = self.predict(features)
            loss = self.loss_fn(pred, labels)
            loss.backward()
            opt.step()
            opt.zero_grad()

    def predict(self, data: np.ndarray):
        return np.zeros((data.shape[0], 1))

    @staticmethod
    def loss_fn(pred, labels):
        return F.cross_entropy(pred, labels)

    @staticmethod
    def accuracy(pred, labels):
        """
        ROC-AUC estimation.

        Args:
            pred: our hypothesis.
            labels: label-vector.

        Returns:
            Accuracy estimation of the prediction.
        """
        pass