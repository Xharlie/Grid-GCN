#!/bin/bash
python3 -u test/test_s3dis_gpu_ggcn.py
python3 -u test/s3dis_merge.py -d ../data/3DIS/prepare_label_rgb/Area_6/
