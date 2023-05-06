import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

class HousePricingModel():

    def __init__(self, n_estimators, max_depth):
        self.model = Pipeline([
            ('encoder', OneHotEncoder()),
            ('model', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth)
            )])

    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        pred = self.model.predict(X)
        return pred

    def save(self,model_name):
        file = open(f"{model_name}.pkl", "wb")
        pickle.dump(self.model, file)

    def load(self,model_name):
        file = open(f"{model_name}.pkl", "rb")
        self.model = pickle.load(file)
