#!/bin/bash
python3 -u test/test_s3dis_gpu_ggcn.py
python3 -u test/s3dis_merge.py -d ../data/3DIS/prepare_label_rgb/Area_5/
# python3 -u test/eval_s3dis.py -d ../data/3DIS/prepare_label_rgb/
