from torch import nn, sigmoid
from torchvision import models

# Simple Logistic Regression model, but we use the version by sklearn instead.
class LR(nn.Module):
    def __init__(self, dims) -> None:
        super().__init__()
        self.logistic_regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dims, 1)
        )

    def forward(self, x, probabilistic):
        h = self.logistic_regression(x)
        if probabilistic:
            h = sigmoid(h)
        return h

# Multilayer perceptron with ReLU activation function for image datasets.
class MLPReLU(nn.Module):
    def __init__(self, dims) -> None:
        super().__init__()
        self.sequential_relu = nn.Sequential(
            nn.Flatten(),
            # bias=False -> normalized after, would be meaningless parameters to learn
            nn.Linear(dims, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            # bias can be on or off for last layer, is just recalibrated by selecting a different threshold
            nn.Linear(300, 1)
        )

    def forward(self, x, probabilistic):
        h = self.sequential_relu(x)
        if probabilistic:
            h = sigmoid(h)
        return h


# Convolutional Neural Network for CIFAR10 dataset.
class FullCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequential_CNN = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1), #output = 32x32
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1), #output = 32x32
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1, stride=2), #output = 16x16
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 192, 3, padding=1), #output = 16x16
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1), #output = 16x16
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1, stride=2), #output = 8x8
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1), #output = 8x8
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1), #output = 8x8
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 10, 1), #output = 8x8
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*8*10, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
        )

    def forward(self, x, probabilistic):
        h = self.sequential_CNN(x)
        if probabilistic:
            h = sigmoid(h)
        return h
    
class Resnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x, probabilistic):
        h = self.model(x)
        if probabilistic:
            h = sigmoid(h)
        return h