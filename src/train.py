from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

FEATURES = ['hour', 'dayofweek', 'month', 'is_weekend', 'is_holiday',
            'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'lag_24h', 'lag_168h']
TARGET = 'traffic_volume'


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute la variable binaire is_holiday à partir de la colonne holiday.
    À appeler après add_time_features et add_lag_features.
    """
    df = df.copy()
    df['is_holiday'] = (df['holiday'] != 'None').astype(int)
    return df


def split_train_test(df: pd.DataFrame, cutoff: str = '2018-04-01'):
    """
    Découpage chronologique : train avant la date de coupure, test à partir de celle-ci.
    """
    cut = pd.Timestamp(cutoff)
    return df[df['date_time'] < cut], df[df['date_time'] >= cut]


def get_model(name: str = 'random_forest', **params):
    """
    Retourne un modèle scikit-learn non entraîné ('linear' ou 'random_forest').
    """
    if name == 'linear':
        return LinearRegression(**params)
    if name == 'random_forest':
        defaults = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    raise ValueError(f"Modèle inconnu : {name}")


def train_model(train: pd.DataFrame, name: str = 'random_forest', features=FEATURES, **params):
    """
    Entraîne le modèle choisi sur le jeu d'entraînement et le retourne.
    """
    model = get_model(name, **params)
    return model.fit(train[features], train[TARGET])