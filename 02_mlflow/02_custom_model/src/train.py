import os
import mlflow
import pandas as pd
from dotenv import load_dotenv
from model.custom_model import GBWrapper

load_dotenv('src/.env.mlflow')

# Leemos los datos
data = pd.read_csv(os.environ['DATASET_PATH'])
X = data.drop(os.environ['TARGET_NAME'],axis=1)
y = data[os.environ['TARGET_NAME']]

# Entrenamos el modelo
model = GBWrapper(
    n_estimators=int(os.environ['N_ESTIMATORS'])
    ).fit(X, y)

# Registramos el modelo desde la función que hemos creado
model.log_model(
    artifact_path=os.environ['MODEL_NAME'],
    registered_model_name=os.environ['MODEL_NAME'])
