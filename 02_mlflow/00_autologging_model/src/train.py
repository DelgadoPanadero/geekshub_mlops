import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression

# mlflow
mlflow.autolog()

#load_dotenv('src/.env.mlflow')
max_iter=100

# Leemos los datos
data = pd.read_csv('data/iris.csv')
X = data.drop('Species',axis=1)
y = data['Species']

# Entrenamos el modelo
model = LogisticRegression(max_iter=max_iter)
model.fit(X, y)
score = model.score(X,y)
