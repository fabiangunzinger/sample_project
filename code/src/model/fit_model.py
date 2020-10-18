from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


def fit(model, features, target):
    """Fit model to training data."""
    model = model.fit(features, target)
    return model


def traincv(model, features, target, cv=5):
    """Train model on data."""
    prediction = cross_val_predict(model, features, target,
                                   cv=cv, method='predict')
    probability = cross_val_predict(model, features, target,
                                    cv=cv, method='predict_proba').T[1]
    return prediction, probability
