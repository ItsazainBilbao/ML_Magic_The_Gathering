# models.py
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class MultiModelRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_cheap, model_mid, model_exp, price_splitter):
        self.model_cheap = model_cheap
        self.model_mid = model_mid
        self.model_exp = model_exp
        self.price_splitter = price_splitter

    def fit(self, X, y):
        y = y.values if hasattr(y, 'values') else y
        mask_cheap, mask_mid, mask_exp = self.price_splitter(y)
        self.model_cheap.fit(X[mask_cheap], y[mask_cheap])
        self.model_mid.fit(X[mask_mid], y[mask_mid])
        self.model_exp.fit(X[mask_exp], y[mask_exp])
        return self

    def predict(self, X):
        preds_cheap = self.model_cheap.predict(X)
        preds_mid = self.model_mid.predict(X)
        preds_exp = self.model_exp.predict(X)
        return np.vstack([preds_cheap, preds_mid, preds_exp]).T
