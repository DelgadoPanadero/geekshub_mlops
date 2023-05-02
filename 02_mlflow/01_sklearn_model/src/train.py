import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression

load_dotenv('src/.env.mlflow')
max_iter=100

# Leemos los datos
data = pd.read_csv('data/iris.csv')
X = data.drop('Species',axis=1)
y = data['Species']

# Entrenamos el modelo
model = LogisticRegression(max_iter=max_iter)
model.fit(X, y)
score = model.score(X,y)

# Logeamos los parametros
mlflow.log_param('max_iter', max_iter)
mlflow.log_metric('score', score)

# Para registrar un modelo nuevo podemos usar
# lod_model definido “registered_model_name”
mlflow.sklearn.log_model(
    artifact_path='iris_sklearn_model',
    sk_model=model,
    registered_model_name='iris_sklearn_model')
