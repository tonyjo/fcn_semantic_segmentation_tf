#!/bin/bash

FILE1=$1
CUDA_VISIBLE_DEVICES=0 python src/test_gpu0.py --config_file=$FILE1

