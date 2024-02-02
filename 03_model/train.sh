#!/bin/bash

python -m src.train \
  --max_iter 100 \
  --data_path ./data/iris.csv \
  --model_path model/iris_model.pkl
