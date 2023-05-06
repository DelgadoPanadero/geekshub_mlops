import os
import json
import joblib
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Species")
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)

    parser.add_argument("--train-dir", type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test-dir", type=str,
        default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR"))

    args, _ = parser.parse_known_args()

    # Leemos los datos
    df_train = pd.read_csv(os.path.join(args.train_dir, 'train.csv'))
    X_train = df_train.loc[:, df_train.columns != args.target].values
    y_train = df_train.loc[:, df_train.columns == args.target].values.ravel()

    # Entrenamiento
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf
    ).fit(X_train, y_train)

    # Testeo
    df_test = pd.read_csv(os.path.join(args.test_dir, 'test.csv'))
    X_test = df_test.loc[:, df_test.columns != args.target].values
    y_test = df_test.loc[:, df_test.columns == args.target].values.ravel()
    print("Accuracy=" + str(accuracy_score(y_test, model.predict(X_test))))

    # Guardamos el modelo
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
