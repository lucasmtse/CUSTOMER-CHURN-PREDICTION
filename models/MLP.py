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


import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(
    model,
    dataloader,
    criterion=None,
    device=None,
    average: str = "binary",
    pos_label: int = 1,
    print_report: bool = True,
    threshold: float = 0.5,
):
    """
    Evaluate a PyTorch classification model on a DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        Modèle déjà entraîné.
    dataloader : DataLoader
        Dataloader (val/test) sur lequel évaluer.
    criterion : loss function ou None
        Par ex. torch.nn.BCELoss, CrossEntropyLoss.
        Si None, la loss n'est pas calculée.
    device : torch.device, str ou None
        "cuda", "cpu", etc. Si None, pas de .to(device).
    average : str
        'binary', 'macro', 'weighted', etc. pour precision/recall/F1.
    pos_label : int
        Classe positive pour le binaire (par ex. 1).
    print_report : bool
        Si True, affiche un classification_report et les métriques.

    Returns
    -------
    metrics : dict
        Dict avec 'loss' (si criterion != None), 'accuracy', 'precision', 'recall', 'f1'.
    """

    model.eval()
    if device is not None:
        model.to(device)

    all_labels = []
    all_preds = []
    running_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            if device is not None:
                inputs = inputs.to(device)
                labels = labels.to(device)

            outputs = model(inputs)
            # ----- gestion de la loss -----
            if criterion is not None:
                loss = criterion(outputs, labels)
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                n_samples += batch_size

            # ----- prédiction (binaire ou multi-classes) -----
            # Cas 1 : sortie (N, 1) → binaire (prob/logits)
            if outputs.dim() == 2 and outputs.size(1) == 1:
                # si le modèle ne contient pas déjà un sigmoid :
                preds = (outputs >= threshold).long().view(-1)
                labels_np = labels.view(-1).cpu().numpy()

            # Cas 2 : sortie (N, C) → multi-classes
            else:
                preds = torch.argmax(outputs, dim=1)
                labels_np = labels.cpu().numpy()

            preds_np = preds.cpu().numpy()

            all_labels.append(labels_np)
            all_preds.append(preds_np)

    # concaténer toutes les batches
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    metrics = {}

    if criterion is not None and n_samples > 0:
        metrics["loss"] = running_loss / n_samples

    metrics["accuracy"] = accuracy_score(all_labels, all_preds)
    metrics["precision"] = precision_score(
        all_labels, all_preds, average=average, pos_label=pos_label
    )
    metrics["recall"] = recall_score(
        all_labels, all_preds, average=average, pos_label=pos_label
    )
    metrics["f1"] = f1_score(
        all_labels, all_preds, average=average, pos_label=pos_label
    )

    if print_report:
        print("Classification report :")
        print(classification_report(all_labels, all_preds))
        print("Confusion matrix :")
        print(confusion_matrix(all_labels, all_preds))
        print("Metrics :")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return metrics

threshold=[0.05+i*0.01 for i in range(30)]
def grid_search_threshold(model, metrics="f1", threshold_list=threshold, dataloader=None, device=None):
    """
    Search for the best threshold to optimize a given metric.

    Parameters
    ----------
    model : torch.nn.Module
        Modèle déjà entraîné.
    metrics : str
        La métrique à optimiser ('accuracy', 'precision', 'recall', 'f1').
    threshold_list : list of float
        Liste des seuils à tester.
    dataloader : DataLoader
        Dataloader (val/test) sur lequel évaluer.
    device : torch.device, str ou None
        "cuda", "cpu", etc. Si None, pas de .to(device).

    Returns
    -------
    best_threshold : float
        Le seuil qui maximise la métrique spécifiée.
    best_metric_value : float
        La valeur maximale de la métrique obtenue.
    """
    best_threshold = None
    best_metric_value = -float("inf")

    for threshold in threshold_list:
        eval_metrics = evaluate_model(
            model,
            dataloader,
            device=device,
            print_report=False,
            threshold=threshold,
        )
        metric_value = eval_metrics[metrics]
        if metric_value is not None and metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold

    return best_threshold, best_metric_value