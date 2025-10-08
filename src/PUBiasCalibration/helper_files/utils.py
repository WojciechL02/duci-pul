import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
import os
from PIL import Image
import torch

# Early stopping function when training neural networks.
class EarlyStopping():
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
    
    def __call__(self, model, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = f"Stopped on {self.counter}"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f"{self.counter}/{self.patience}"
        return False

# Label transformer for MNIST data: even vs odd.
class label_transform_MNIST():
    def __init__(self) -> None:
        pass

    def __call__(self, target):
        if target % 2 == 0:
            return 1
        return 0

class label_transform_Alzheimer():
    def __init__(self) -> None:
        pass

    def __call__(self, target):
        if target in [0, 1, 3]:
            return 1
        return 0

# Label transformer for USPS data: digits 0-4 vs 5-9.
class label_transform_USPS():
    def __init__(self) -> None:
        pass
        
    def __call__(self, target):

        if target in [0, 1, 2, 3, 4]:
            return 1
        return 0

# Label transformer for Fashion-MNIST data: top vs bottom.
class label_transform_Fashion():
    def __init__(self) -> None:
        pass
    def __call__(self, target):
        if target in [0, 2, 3, 4, 6]:
            return 1
        return 0

# Label transformer for CIFAR-10 data: transport vs animals.
class label_transform_CIFAR10():
    def __init__(self) -> None:
        pass
    def __call__(self, target):
        if target in [0, 1, 8, 9]:
            return 1
        return 0
    
def make_binary_class(y):
    if np.unique(y).shape[0]>2:
        values, counts = np.unique(y, return_counts=True)
        ind = np.argmax(counts)
        major_class = values[ind]
        for i in np.arange(y.shape[0]):
            if y[i]==major_class:
                y[i]=1
            else:
                y[i]=0
    return y

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None, target_transform=None):
        """
        Args:
            data_folder (str): Path to the dataset folder.
            transform (callable, optional): A function/transform to apply to the images.
            target_transform (callable, optional): A function/transform to apply to the labels.
        """
        self.data_folder = data_folder
        self.transform = transform
        self.target_transform = target_transform
        self.classes = sorted(os.listdir(data_folder))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.img_paths = []
        self.labels = []
        self.images = []

        # Collect image paths and labels
        for cls in self.classes:
            cls_path = os.path.join(data_folder, cls)
            if os.path.isdir(cls_path):
                for file_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, file_name)
                    self.img_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls])

        for img_path in self.img_paths:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            self.images.append(image)
        # self.images = torch.tensor(self.images, dtype=torch.float32)
        # Convert labels to tensor
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for consistency

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    @property
    def data(self):
        """ Return images as a tensor. """
        
        return self.images

    @property
    def targets(self):
        """ Return labels. """
        return self.labels
    