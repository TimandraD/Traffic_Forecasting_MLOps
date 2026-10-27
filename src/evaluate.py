from __future__ import annotations
import numpy as np
import pandas as pd


def evaluate(y_true, y_pred) -> dict:
    """Calcule MAE, RMSE, R² et WAPE (erreur en % du volume total)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred

    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    r2 = 1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    wape = np.sum(np.abs(err)) / np.sum(np.abs(y_true)) * 100

    return {"MAE": float(round(mae, 1)), "RMSE": float(round(rmse, 1)),
            "R2": float(round(r2, 3)), "WAPE_%": float(round(wape, 1))}


def baseline_scores(df: pd.DataFrame, target_col: str = "traffic_volume") -> dict:
    """Score de référence : prédire le trafic de la veille à la même heure (lag_24h)."""
    return evaluate(df[target_col], df["lag_24h"])


def cross_validate_ts(model, df, features, target="traffic_volume", n_splits=5, test_size=24 * 90):
    """Validation croisée temporelle : on entraîne toujours sur le passé et on teste sur la période suivante."""
    from sklearn.base import clone
    from sklearn.model_selection import TimeSeriesSplit

    resultats = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits, test_size=test_size).split(df):
        m = clone(model).fit(df.iloc[train_idx][features], df.iloc[train_idx][target])
        resultats.append(evaluate(df.iloc[test_idx][target], m.predict(df.iloc[test_idx][features])))
    return pd.DataFrame(resultats)