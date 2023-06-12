#!/bin/bash
rm -rf ./runs/*
rm -rf ./logs/*
rm -rf ./data/*
pkill python

export CUDA_VISIBLE_DEVICES=1
max=10
for i in `seq 1 $max`
do
    nohup python -W ignore BM1.py --seeds $i >./logs/log1.txt & 
    nohup python -W ignore BM2.py --seeds $i >./logs/log2.txt & 
    nohup python -W ignore BM3.py --seeds $i >./logs/log3.txt & 
    nohup python -W ignore BM4.py --seeds $i >./logs/log4.txt & 
    nohup python -W ignore BM5.py --seeds $i >./logs/log5.txt & 
    nohup python -W ignore main.py --seeds $i >./logs/log.txt & 
    sleep 10m
done

python -W ignore plotfunc.py
