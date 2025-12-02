import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Classifier(nn.Module):
    def forward(self, x):
        pass


class MLPClassifier(Classifier):
    def __init__(self, input_dim, hidden_dim=10):
        super(MLPClassifier, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x, sigmoid=False):
        h = self.h(x)
        if sigmoid:
            h = torch.sigmoid(h)
        return h


class LogisticClassifier(Classifier):
    def __init__(self, input_dim):
        super(LogisticClassifier, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(input_dim, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x, sigmoid=False):
        h = self.h(x)
        if sigmoid:
            h = torch.sigmoid(h)
        return h


class PropensityEstimator(nn.Module):
    def forward(self, x):
        pass


class MLPPropensityEstimator(PropensityEstimator):
    def __init__(self, input_dim, hidden_dim=10):
        super(MLPPropensityEstimator, self).__init__()
        self.eta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid(),
        )

        for layer in [
            module for module in self.eta.modules() if isinstance(module, nn.Linear)
        ]:
            # layer.weight = nn.Parameter(torch.randn_like(layer.weight) / 100)
            # layer.bias = nn.Parameter(torch.randn_like(layer.bias) / 100)
            layer.weight = nn.Parameter(torch.zeros_like(layer.weight))
            layer.bias = nn.Parameter(torch.zeros_like(layer.bias))

    def forward(self, x, sigmoid=False):
        eta = self.eta(x)
        if sigmoid:
            eta = torch.sigmoid(eta)
        return eta


class LogisticPropensityEstimator(PropensityEstimator):
    def __init__(self, input_dim):
        super(LogisticPropensityEstimator, self).__init__()
        self.eta = nn.Sequential(
            nn.Linear(input_dim, 1),
            # nn.Sigmoid(),
        )

        for layer in [
            module for module in self.eta.modules() if isinstance(module, nn.Linear)
        ]:
            # layer.weight = nn.Parameter(torch.randn_like(layer.weight) / 100)
            # layer.bias = nn.Parameter(torch.randn_like(layer.bias) / 100)
            layer.weight = nn.Parameter(torch.zeros_like(layer.weight))
            layer.bias = nn.Parameter(torch.zeros_like(layer.bias))

    def forward(self, x, sigmoid=False):
        eta = self.eta(x)
        if sigmoid:
            eta = torch.sigmoid(eta)
        return eta


class LBE(nn.Module):
    def __init__(self, input_dim, kind="MLP", hidden_dim=10, device=None):
        super(LBE, self).__init__()
        # Determine device
        if device is None:
            self.device = (
                "mps"
                if getattr(torch, "has_mps", False)
                else "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        if kind == "MLP":
            self.h = MLPClassifier(input_dim, hidden_dim)
            self.eta = MLPPropensityEstimator(input_dim, hidden_dim)
        elif kind == "LR":
            self.h = LogisticClassifier(input_dim)
            self.eta = LogisticPropensityEstimator(input_dim)

        # Move model to device
        self.to(self.device)

    def get_theta_h(self):
        value = torch.tensor([], dtype=torch.float, device=self.device)
        for param in self.h.parameters():
            value = torch.cat([value.squeeze(), param.data.reshape(-1)])
        return value

    def get_theta_eta(self):
        value = torch.tensor([], dtype=torch.float, device=self.device)
        for param in self.eta.parameters():
            value = torch.cat([value.squeeze(), param.data.reshape(-1)])
        return value

    def forward(self, x):
        # Move input tensor to device
        x = x.to(self.device)
        h = self.h(x, sigmoid=True)
        return h

    def E_step(self, x, s):
        with torch.no_grad():
            # Move input tensors to device
            x = x.to(self.device)
            s = s.to(self.device)

            h = self.h(x, sigmoid=True).squeeze()
            eta = self.eta(x, sigmoid=True).squeeze()

            P_y_hat_1 = torch.where(s == 1, eta, 1 - eta) * h
            P_y_hat_0 = torch.where(s == 1, 0, 1) * (1 - h)

            P_y_hat = torch.cat(
                [P_y_hat_0.reshape(-1, 1), P_y_hat_1.reshape(-1, 1)], axis=1
            )
            P_y_hat /= P_y_hat.sum(axis=1).reshape(-1, 1)
            return P_y_hat

    def loss(self, x, s, P_y_hat):
        # Move input tensors to device
        x = x.to(self.device)
        s = s.to(self.device)
        P_y_hat = P_y_hat.to(self.device)

        h = self.h(x).squeeze()
        eta = self.eta(x).squeeze()

        log_h = F.logsigmoid(h)
        log_1_minus_h = F.logsigmoid(-h)
        log_eta = F.logsigmoid(eta)
        log_1_minus_eta = F.logsigmoid(-eta)

        # loss = torch.where(
        #     s == 1,
        #     P_y_hat[:, 1] * (log_h + log_eta) + 0,
        #     P_y_hat[:, 1] * (log_h + log_1_minus_eta) + P_y_hat[:, 0] * log_1_minus_h
        # )
        loss = torch.where(
            s == 1,
            P_y_hat[:, 1] * (log_h + log_eta)
            + P_y_hat[:, 0] * (log_1_minus_h + log_eta),
            P_y_hat[:, 1] * (log_h + log_1_minus_eta)
            + P_y_hat[:, 0] * (log_1_minus_h + log_1_minus_eta),
        )

        with torch.no_grad():
            x_ones = torch.ones((len(x), 1), device=self.device)
            x = torch.cat([x, x_ones], dim=1)
            sigma_h = torch.sigmoid(h)
            sigma_eta = torch.sigmoid(eta)
            grad_theta_1 = (
                (P_y_hat[:, 0] * sigma_h).reshape(-1, 1) * x
                + (P_y_hat[:, 1] * (sigma_h - 1)).reshape(-1, 1) * x
            ).sum(axis=0)
            grad_theta_2 = (
                (
                    (-1) ** (s + 1)
                    * (
                        1
                        * P_y_hat[:, 1]
                        / torch.where(s == 1, sigma_eta, 1 - sigma_eta)
                    )
                    * sigma_eta
                    * (sigma_eta - 1)
                ).reshape(-1, 1)
                * x
            ).sum(axis=0)

        return -torch.sum(loss), grad_theta_1, grad_theta_2

    def pre_train(self, x, s, epochs=100, lr=1e-3, print_msg=False):
        # Move input tensors to device
        x = x.to(self.device)
        s = s.to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.h.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            s_logits = self.h(x)
            # Compute Loss
            loss = criterion(s_logits.squeeze(), s)
            # Backward pass
            loss.backward()
            optimizer.step()

            if print_msg:
                print("Epoch {}: train loss: {}".format(epoch, loss.item()))


def lbe_train(X, s, kind="LR", epochs=1000, device=None):
    """
    Train an LBE model with optional GPU acceleration.

    Args:
        X: Input features (numpy array)
        s: Labels (numpy array)
        kind: Type of model ('LR' or 'MLP')
        epochs: Number of training epochs
        device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detection)

    Returns:
        Trained LBE model
    """
    p = X.shape[1]
    lbe = LBE(p, kind=kind, device=device)

    # Print device information
    # print(f"Training LBE model on device: {lbe.device}")

    X = torch.from_numpy(X)
    s = torch.from_numpy(s)

    X = X.float()
    s = s.float()

    # Pre-training
    lbe.pre_train(X, s, epochs=1000, lr=1e-2)

    optimizer = optim.Adam(lbe.parameters(), lr=1e-2)

    # Main training loop
    epochs = 100
    for epoch in range(epochs):
        P_y_hat = lbe.E_step(X, s)

        for M_step_iter in range(100):
            optimizer.zero_grad()
            loss, grad_theta_1, grad_theta_2 = lbe.loss(X, s, P_y_hat)
            loss.backward()
            optimizer.step()
    return lbe


def lbe_predict_proba(lbe, Xtest):
    """
    Get probability predictions from a trained LBE model.

    Args:
        lbe: Trained LBE model
        Xtest: Test features (numpy array)

    Returns:
        Probability predictions (numpy array)
    """

    Xtest = torch.from_numpy(Xtest)
    Xtest = Xtest.float()

    # Forward pass through the model (model.forward will move to device)
    lbe_out = lbe(Xtest)

    # Move results back to CPU for numpy conversion
    y_proba = lbe_out.squeeze().detach().cpu().numpy()
    return y_proba
