import joblib
import os
import pandas as pd
from src import config


def create_idx_file(file_name, index, replace=False):
    """Create indexed csv file."""
    path = config.DATADIR
    filepath = os.path.join(path, file_name)
    if (os.path.isfile(filepath)) and (not replace):
        return f'{file_name} already exists.'
    pd.DataFrame(index=index).to_csv(filepath)
    return f'{file_name} created.'


def store_to_file(file_name, series, col_name, replace=False):
    """Add series to file."""
    path = config.DATADIR
    filepath = os.path.join(path, file_name)
    df = pd.read_csv(filepath)
    if (col_name in df) and (not replace):
        return f'{col_name} already in {file_name}. Not added.'
    df[col_name] = series
    df.to_csv(filepath, index=False)
    return f'{col_name} added to {file_name}.'


def store_model(model, model_name, replace=False):
    """Store model."""
    path = config.MODELDIR
    filepath = os.path.join(path, model_name)
    if (os.path.isfile(filepath)) and (not replace):
        return f'{model_name} already exists.'
    joblib.dump(model, filepath)
    return f'{model_name} stored.'
