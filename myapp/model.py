import os
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),

            nn.Linear(in_features=6272, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_classes)
        )

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            output = self(x)
            predictions = torch.argmax(output, dim=1)

        return predictions
    def forward(self,x):
        return self.model(x)