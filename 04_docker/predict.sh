#!/bin/bash

python -m src.predict \
  --model_path model/iris_model.pkl \
  --port 8888
