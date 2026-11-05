import torch
import torch.nn as nn
import numpy as np


def train_model(
    model,
    inputs,
    labels,
    num_epochs,
    criterion=nn.MSELoss(),
    optimizer=torch.optim.Adam,
    print_every: int = 50,
):
    """
    Function to train a given sequence model.
    Args:
        model : torch.nn.Module
            sequence model
        inputs : torch.Tensor
            input data
        num_epochs : int
            number of epochs
        labels : torch.Tensor
            labels
        criterion : torch.nn.Module
            loss function
        optimizer : torch.optim.Optimizer
            optimizer
    Returns:
        None
    """

    hist = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        # Enforce sample-wise aligned loss (avoid accidental broadcasting like (N,1) vs (N,))
        pred = outputs.reshape(-1)
        target = labels.to(device=pred.device, dtype=pred.dtype).reshape(-1)
        if pred.numel() != target.numel():
            raise ValueError(
                f"Prediction/target size mismatch after flatten: pred={pred.shape}, target={target.shape}"
            )
        loss = criterion(pred, target)
        hist[epoch] = loss.item()
        loss.backward()
        optimizer.step()
        
        if epoch % max(1, int(print_every)) == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch} loss: {loss.item()}')
        
    
    print(f"Finished Training. Final loss: {loss.item()}")

    return None


# Backward-compatible alias
train_lstm = train_model
