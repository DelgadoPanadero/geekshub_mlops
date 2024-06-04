import os
import mlflow
import pandas as pd
from model.custom_model import GBWrapper


# Leemos los datos
data = pd.read_csv('data/iris.csv')
X = data.drop('Species',axis=1)
y = data['Species']

# Entrenamos el modelo
model = GBWrapper(
    n_estimators=int(100)
    ).fit(X, y)

# Registramos el modelo desde la función que hemos creado
model.log_model(
    artifact_path="iris_custom_model",
    registered_model_name="iris_custom_model")
