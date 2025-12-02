# utils/evaluation.py

"""
Utility functions for model evaluation.

- evaluate_model : compute main metrics on a test set
- cross_validate_model : perform cross-validation with several metrics
- plot_confusion_matrix : confusion matrix plot
- plot_roc_curve : ROC curve plot (binary classification)
"""

from typing import Dict, Any, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.base import BaseEstimator


def evaluate_model(
    model: BaseEstimator,
    X_test,
    y_test,
    average: str = "binary",
    pos_label: Union[int, str] = 1,
    print_report: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a fitted classification model on a test set.

    Parameters
    ----------
    model : estimator
        Fitted sklearn-like model with a .predict() method.
    X_test : array-like
        Test features.
    y_test : array-like
        True labels for the test set.
    average : str
        Averaging method for precision/recall/F1 ('binary', 'macro', 'weighted', ...).
    pos_label : int or str
        Label considered as positive class for binary classification.
    print_report : bool
        If True, prints a sklearn classification_report.

    Returns
    -------
    metrics : dict
        Dictionary with accuracy, precision, recall, f1 (+ roc_auc if available).
    """
    y_pred = model.predict(X_test)

    metrics: Dict[str, float] = {}

    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    metrics["precision"] = precision_score(
        y_test, y_pred, average=average, pos_label=pos_label
    )
    metrics["recall"] = recall_score(
        y_test, y_pred, average=average, pos_label=pos_label
    )
    metrics["f1"] = f1_score(
        y_test, y_pred, average=average, pos_label=pos_label
    )

    # Try to compute ROC AUC if probabilities or decision function exist
    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            # binary: use proba of positive class
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                y_score = y_score[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_score)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            metrics["roc_auc"] = roc_auc_score(y_test, y_score)
    except Exception:
        # silently ignore if ROC AUC cannot be computed
        pass

    if print_report:
        print("Classification report:")
        print(classification_report(y_test, y_pred))

        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return metrics


def cross_validate_model(
    estimator: BaseEstimator,
    X,
    y,
    cv: int = 5,
    scoring: Optional[Union[str, Sequence[str]]] = None,
    n_jobs: int = -1,
    random_state: int = 42,
    return_estimator: bool = False,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Perform cross-validation on a (non-fitted) estimator.

    Parameters
    ----------
    estimator : estimator
        Unfitted sklearn-like model (DecisionTreeClassifier, XGBClassifier, etc.).
    X : array-like
        Features.
    y : array-like
        Labels.
    cv : int
        Number of folds (StratifiedKFold if classification).
    scoring : str or list of str, optional
        Scoring metrics for cross_validate. If None, defaults to
        ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'].
    n_jobs : int
        Number of parallel jobs for cross_validate.
    random_state : int
        Random seed for StratifiedKFold.
    return_estimator : bool
        If True, returns the estimators fitted on each fold.
    verbose : int
        Verbosity for cross_validate.

    Returns
    -------
    cv_results : dict
        cross_validate results dict (fit_time, score_time, test_<metric>, ...).
    """
    if scoring is None:
        scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    cv_splitter = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )

    results = cross_validate(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv_splitter,
        n_jobs=n_jobs,
        return_estimator=return_estimator,
        verbose=verbose,
    )

    # Petit résumé rapide dans la console
    print("Cross-validation results:")
    for key in results:
        if key.startswith("test_"):
            scores = results[key]
            print(f"{key}: mean={scores.mean():.4f}  std={scores.std():.4f}")

    return results


def plot_confusion_matrix(
    model: BaseEstimator,
    X_test,
    y_test,
    labels: Optional[Sequence] = None,
    normalize: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "Blues",
):
    """
    Plot confusion matrix of a fitted model.

    Parameters
    ----------
    model : estimator
        Fitted model with .predict().
    X_test : array-like
        Test features.
    y_test : array-like
        True labels.
    labels : list, optional
        List of label names (order of rows/cols).
    normalize : {'true', 'pred', 'all', None}, optional
        Normalization mode for confusion_matrix.
    title : str, optional
        Plot title.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    y_pred = model.predict(X_test)

    if labels is None:
        labels = np.unique(y_test)

    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize=normalize)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=cmap, colorbar=True)

    if title is None:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix"
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_roc_curve(
    model: BaseEstimator,
    X_test,
    y_test,
    pos_label: Union[int, str] = 1,
    title: str = "ROC curve",
):
    """
    Plot ROC curve for a binary classifier.

    Parameters
    ----------
    model : estimator
        Fitted model with predict_proba or decision_function.
    X_test : array-like
        Test features.
    y_test : array-like
        True labels.
    pos_label : int or str
        Positive class label.
    title : str
        Plot title.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise ValueError(
            "Model must have predict_proba or decision_function to plot ROC curve."
        )

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_test, y_score, pos_label=pos_label, ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax
