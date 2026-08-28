#!/bin/bash

pip install -r requirements.txt
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
