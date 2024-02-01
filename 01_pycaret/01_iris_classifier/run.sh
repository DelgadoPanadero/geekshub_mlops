#!/bin/bash

sudo docker run -it -v ./:/home -p 8888:8888 -p 8000:8000 python:3.10 bash
