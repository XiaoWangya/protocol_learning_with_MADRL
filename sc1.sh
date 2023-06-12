pkill python
python move_his.py
max=10
for i in `seq 1 $max`
do  
    for j in `seq 6 10`
    do
        CUDA_VISIBLE_DEVICES=1 nohup python -W ignore main.py --n_client $j >./logs/log.txt & 
        CUDA_VISIBLE_DEVICES=0 nohup python -W ignore BM5.py --n_client $j >./logs/log5.txt & 
        sleep 10m
    done
done
