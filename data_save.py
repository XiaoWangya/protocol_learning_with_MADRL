import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, re
from copy import deepcopy
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
sns.set()

# #加载日志数据
def smooth(datalist, weight:float=0.25, block:int = 500):
    new_datalist = []
    for item in datalist:
        new_datalist.append(deepcopy(weight * np.mean(datalist[max(datalist.index(item)-block, 0): datalist.index(item)]) + item*(1-weight)))
    return new_datalist

def main(dirpath:str='./runs/'):
    event_dir = dirpath
    df = pd.DataFrame()
    all_dir = os.listdir(event_dir)
    all_dir.sort()
    for file in all_dir:
        temp_df = pd.DataFrame()
        subdir = event_dir+file + '/'
        type = re.search(r'_[a-zA-Z]{2,4}\d{0,2}', subdir).group(0)[1:]
        ea_name = [item for item in os.listdir(subdir) if re.match(r'events*', item) is not None][0]
        ea=event_accumulator.EventAccumulator(subdir + ea_name).Reload().scalars
        key_dir = ea.Keys()
        temp_df['step'] = [item.step for item in ea.Items(key_dir[0])]
        for key in key_dir[:8]:
            temp_df[key] = [item.value for item in ea.Items(key)]
        temp_df['method'] = type
        temp_df['Number of edge devices'] = ea.Items(key_dir[-1])[0].value
        df = pd.concat([df, temp_df])
    #save file
    df.to_csv('./data/saved.csv')
main()