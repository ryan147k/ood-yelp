#!/bin/bash
# 使用ex_num控制main函数里调用哪一个实验
ex_num=2
CUDA_VISIBLE_DEVICES=7 nohup python -u main.py --ex_num=${ex_num} > log/ex${ex_num}.log 2>&1 &
