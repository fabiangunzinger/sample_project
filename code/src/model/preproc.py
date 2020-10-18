import numpy as np

from zlib import crc32
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..features.decorators import preproc_registry


def split_train_test(data, test_ratio):
    """Split data into training and test set.

    Test ratio determines proportion of users allocated to test set.

    Split relies on hash function to ensure that allocation of
    a single user is independent of total set of users in the
    data.
    """
    def test_set_check(identifier, test_ratio):
        return (crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32)

    id_list = data.index.to_series()
    in_test_set = id_list.apply(lambda id: test_set_check(id, test_ratio))

    test_set = data.loc[in_test_set]
    train_set = data.loc[~in_test_set]
    return train_set, test_set


def split_features_target(df, target_name):
    """Split features and target into separate tables."""
    target = df.loc[:, target_name]
    features = df.drop(target_name, axis=1)
    return features, target


def prepare_target(target):
    """Return target vector ready for ml-modeling."""
    le = LabelEncoder().fit(target)

    return le.transform(target), le.classes_


def prepare_features(features, preproc=None):
    """Return features ready for ml-modeling."""
    if preproc is None:
        preproc = preproc_registry

    pipeline = make_pipeline(
        StandardScaler(),
    )

    full_pipeline = make_column_transformer(
        (pipeline, preproc),
        remainder='passthrough'
    )

    return (
        full_pipeline.fit_transform(features),
        features.columns,
        full_pipeline
    )
