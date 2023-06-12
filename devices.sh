# rm -rf ./runs/*
# rm -rf ./logs/*
# rm -rf ./data/*
pkill python
python move_his.py

nohup sh sc1.sh >.log.txt &
nohup sh sc2.sh >.log2.txt &

python -W ignore data_save.py
