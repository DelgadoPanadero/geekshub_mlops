import logging

import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class CustomModel():

    def __init__(self, max_iter: int = 100):

        self._features = []
        self._max_iter = max_iter
        self._is_fitted = False

        self._processor = StandardScaler()
        self._predictor = LogisticRegression(max_iter=max_iter)


    @property
    def max_iter(self):
        return self._max_iter


    @property
    def is_fitted(self):
        return self._is_fitted


    @property
    def features(self):
        return self._features


    def fit(self, X: pd.DataFrame, y: pd.Series):

        _X = self._processor.fit_transform(X)
        self._predictor.fit(_X,y)

        self._features = set(X.columns.to_list())
        self._is_fitted = True

        return self


    def predict(self, X: pd.DataFrame):

        X = self._processor.transform(X)
        y = self._predictor.predict(X)

        return y


    def score(self, X: pd.DataFrame, y: pd.DataFrame):

        return self._predictor.score(
            self._processor.transform(X), y
            )


    def save(self, model_path: str):

        with open(model_path, "wb") as fout:
            pickle.dump(self,fout)
        return self


    def load(self, model_path: str):

        with open(model_path, "rb") as fin:
            self = pickle.load(fin)
        return self
