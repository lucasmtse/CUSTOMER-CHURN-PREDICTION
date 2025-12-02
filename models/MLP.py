import torch 
from torch import nn
from torch.nn import functional as F

class MLP_classifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super(MLP_classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 64)
        self.act2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs=25,
    val_loader=None,
    checkpoint_path="checkpoint/best_model.pt",
    device=None,
):
    """
    Train a PyTorch model with optional validation and checkpointing.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_loader : DataLoader
        Dataloader for training data.
    criterion : loss function
        e.g. torch.nn.BCELoss, CrossEntropyLoss, etc.
    optimizer : torch.optim.Optimizer
    num_epochs : int
    val_loader : DataLoader or None
        If provided, used to compute validation loss every epoch.
    checkpoint_path : str
        Where to save the best model (based on lowest val_loss).
    device : torch.device or str or None
        If not None, model and data are moved to this device.
    """

    if device is not None:
        model.to(device)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Train 
        running_loss = 0.0
        for inputs, labels in train_loader:
            if device is not None:
                inputs = inputs.to(device)
                labels = labels.to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        epoch_val_loss = None
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    if device is not None:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)

            epoch_val_loss = val_running_loss / len(val_loader.dataset)

            # Checkpoint : on sauvegarde si la val_loss s'AMÉLIORE (diminue)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    checkpoint_path,
                )
                print(
                    f"✅ Checkpoint saved at epoch {epoch+1} "
                    f"(val_loss improved to {best_val_loss:.4f})"
                )

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:  
            if epoch_val_loss is not None:
                print(
                    f"Epoch {epoch+1}/{num_epochs} "
                    f"- train_loss: {epoch_train_loss:.4f} "
                    f"- val_loss: {epoch_val_loss:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{num_epochs} "
                    f"- train_loss: {epoch_train_loss:.4f}"
                )

    return model


def evaluate_model(model, dataloader):
    """
    Evaluate a PyTorch model on a given dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to evaluate.
    dataloader : DataLoader
        Dataloader for evaluation data.

    Returns
    -------
    all_outputs : torch.Tensor
        Model outputs for all samples in the dataloader.
    all_labels : torch.Tensor
        True labels for all samples in the dataloader.
    """
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_outputs, all_labels