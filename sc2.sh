max=10

 for i in `seq 1 $max`
do
    for j in `seq 6 10`
    do
        CUDA_VISIBLE_DEVICES=2 nohup python -W ignore BM1.py --n_client $j >./logs/log1.txt & 
        CUDA_VISIBLE_DEVICES=2 nohup python -W ignore BM2.py --n_client $j >./logs/log2.txt & 
        CUDA_VISIBLE_DEVICES=3 nohup python -W ignore BM3.py --n_client $j >./logs/log3.txt & 
        CUDA_VISIBLE_DEVICES=3 nohup python -W ignore BM4.py  --n_client $j >./logs/log4.txt & 
        sleep 8m
    done
done