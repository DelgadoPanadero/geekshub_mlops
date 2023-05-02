import os
import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv('src/.env.mlflow')

# Seleccionamos el modelo
name = os.environ['MODEL_NAME']
version = 1
uri = f"models:/{name}/{version}"

# Leemos los datos
data = pd.read_csv(os.environ['DATASET_PATH'])
X = data.drop(os.environ['TARGET_NAME'],axis=1)
y = data[os.environ['TARGET_NAME']]

# Podemos obtener una referencia al modelo a # partir de su URI
model = mlflow.pyfunc.load_model(model_uri=uri)

# Llamamos al modelo como si fuese un modelo
# de sklearn, sin embargo, esto solo realiza una
# llamada al server. Las predicciones se hacen
# en el server
pred = model.predict(X)
print(pred)
