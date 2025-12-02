from typing import Literal, Any
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

ModelType = Literal['random_forest', 'logistic_regression', 'xgboost']

def train(
    X_train,
    y_train,
    model_type: ModelType = 'xgboost',
    random_state: int = 42,
    n_estimators: int = 100,
    **kwargs: Any
):
    """
    Train a classification model.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    model_type : {'decision_tree', 'logistic_regression', 'xgboost'}
        Type of model to train.
    random_state : int
        Random seed (used when applicable).
    n_estimators : int
        Number of estimators (for XGBoost).
    **kwargs : dict
        Extra parameters passed to the underlying model.

    Returns
    -------
    model
        Fitted model instance.
    """

    if model_type == 'random_forest':
        model = RandomForestClassifier(class_weight="balanced",
            random_state=random_state,
            **kwargs
        )

    elif model_type == 'logistic_regression':
        # max_iter augmenté pour éviter les warnings de convergence
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            **kwargs
        )

    elif model_type == 'xgboost':
        model = XGBClassifier(scale_pos_weight=3.5,
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    return model
