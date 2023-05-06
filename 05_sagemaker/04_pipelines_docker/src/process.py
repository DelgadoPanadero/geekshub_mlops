import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


base_dir = "/opt/ml/processing"

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Species")
    parser.add_argument("--input-file", type=str, default="iris.csv")
    parser.add_argument("--input-dir", type=str, default=base_dir+"/input")
    parser.add_argument("--output-dir", type=str, default=base_dir+"/output")
    args, _ = parser.parse_known_args()

    # Read data
    df = pd.read_csv(os.path.join(args.input_dir, args.input_file))
    X = df.loc[:, df.columns != args.target]
    y = df.loc[:, df.columns == args.target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=28)

    # Save data
    pd.concat([X_train, y_train], axis=1).to_csv(
        os.path.join(args.output_dir, "train/train.csv"),
        index=False)

    pd.concat([X_test, y_test], axis=1).to_csv(
        os.path.join(args.output_dir, "test/test.csv"),
        index=False)
