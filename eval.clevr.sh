#!/bin/bash

python run_network.py --eval --gpus 0 --expname clevr-exp --dataset clevr --pretrained-pkl gdrive:clevr-snapshot.pkl
