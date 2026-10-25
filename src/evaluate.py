from __future__ import annotations
import numpy as np
import pandas as pd


def evaluate(y_true, y_pred) -> dict:
    """Calcule les métriques de régression : MAE, RMSE et R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {"MAE": float(round(mae, 1)), "RMSE": float(round(rmse, 1)), "R2": float(round(r2, 3))}


def baseline_scores(df: pd.DataFrame, target_col: str = "traffic_volume") -> dict:
    """Score de référence : prédire le trafic de la veille à la même heure (lag_24h)."""
    return evaluate(df[target_col], df["lag_24h"])