from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def load_and_clean_data(file_path: str | Path) -> pd.DataFrame:
    """
    Charge le fichier CSV du trafic, convertit la colonne date_time,
    trie les données par ordre chronologique et supprime les doublons.
    """
    df = pd.read_csv(file_path)

    # Conversion de la date en datetime
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Tri chronologique indispensable pour une série temporelle
    df = df.sort_values('date_time', kind='stable')

    # Suppression des doublons sur le timestamp
    df = df.drop_duplicates(subset=['date_time']).reset_index(drop=True)

    # "None" est lu comme NaN par pandas : on le remet comme vraie valeur
    df['holiday'] = df['holiday'].fillna('None')

    # Valeurs aberrantes -> NaN (temp à 0 K, rain_1h à 9831 mm)
    df.loc[df['temp'] < 200, 'temp'] = np.nan
    df.loc[df['rain_1h'] > 100, 'rain_1h'] = np.nan

    # On garde la plus longue période continue (après le gros trou de ~10 mois)
    gap = df['date_time'].diff() > pd.Timedelta(days=30)
    if gap.any():
        df = df[df['date_time'] >= df.loc[gap, 'date_time'].iloc[-1]]

    # Grille horaire régulière + interpolation des petits trous (6 h max)
    df = df.set_index('date_time').asfreq('h')
    num_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']
    df[num_cols] = df[num_cols].interpolate(limit=6)
    df['holiday'] = df['holiday'].fillna('None')
    df['weather_main'] = df['weather_main'].ffill(limit=6)
    df['weather_description'] = df['weather_description'].ffill(limit=6)

    return df.reset_index()