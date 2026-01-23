import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    lbe.pre_train(X, s, epochs=100, lr=1e-3)

    optimizer = optim.Adam(lbe.parameters(), lr=1e-4)

    # Main training loop
    epochs = 50
    for epoch in range(epochs):
        P_y_hat = lbe.E_step(X, s)

        for M_step_iter in range(20):
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
