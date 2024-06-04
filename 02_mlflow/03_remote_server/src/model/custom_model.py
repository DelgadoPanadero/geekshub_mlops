import tempfile

import joblib
import mlflow
from sklearn.ensemble import GradientBoostingClassifier


class GBWrapper(mlflow.pyfunc.PythonModel):


    def __init__(self, n_estimators=100):

        self._n_estimators = n_estimators

        self._model = GradientBoostingClassifier(
            n_estimators = n_estimators
            )

    @property
    def n_estimators(self):
        return self._n_estimators


    @property
    def model(self):
        return self._model


    def load_context(self, context):
        self._model = joblib.load(context.artifacts["model"])


    def predict(self, context, model_input):
        return self._model.predict(
            model_input.values)


    def fit(self, X, y):
        self._model.fit(X,y)
        self.signature=mlflow.models.signature.infer_signature(X,y)

        mlflow.log_param('n_estimators', self.n_estimators)
        mlflow.log_metric('score', self.model.score(X,y))

        return self


    def score(self, X, y):
        return self._model.score(X,y)


    def log_model(self, artifact_path, registered_model_name):

        with tempfile.TemporaryDirectory() as tmpdirname:
            joblib.dump(self._model,f"{tmpdirname}/model.joblib")

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=self,
                artifacts={"model": f"{tmpdirname}/model.joblib"},
                registered_model_name=registered_model_name,
                signature=self.signature,
                conda_env={
                    "channels": ["defaults"],
                    "dependencies": [
                        "python=3.8.10",
                        "pip",
                        {
                            "pip": [
                                "mlflow",
                                "scikit-learn==1.2.2",
                                "cloudpickle==2.2.1",
                            ],
                        },
                    ],
                    "name": "gb_env",
                },
            )

        return self
