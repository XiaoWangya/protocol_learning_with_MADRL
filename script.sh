
rm -rf ./runs/*
rm -rf ./logs/*
pkill python

nohup python -W ignore BM1.py >./logs/log1.txt & 
nohup python -W ignore BM2.py >./logs/log2.txt & 
nohup python -W ignore BM3.py >./logs/log3.txt & 
nohup python -W ignore BM4.py >./logs/log4.txt & 
nohup python -W ignore BM5.py >./logs/log5.txt & 
nohup python -W ignore main.py >./logs/log.txt & 