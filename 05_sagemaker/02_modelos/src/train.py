import os
import argparse

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Species")
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)

    parser.add_argument("--train-file", type=str, default="iris.csv")
    parser.add_argument("--train-dir", type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR"))

    args, _ = parser.parse_known_args()

    df = pd.read_csv(
        os.path.join(args.train_dir, args.train_file)
    )

    X = df.loc[:, df.columns != args.target].values
    y = df.loc[:, df.columns == args.target].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=28)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf
    ).fit(X_train, y_train)
    print("Accuracy=" + str(accuracy_score(y_test, model.predict(X_test))))

    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
