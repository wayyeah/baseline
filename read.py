import numpy as np
import pickle
import matplotlib.pyplot as plt

path='/home/xmu/projects/xmuda/baseline/data/xmu/pseudo_gt_database_info_ouster2hesai_ours.pkl'
with open(path, 'rb') as f:
    info=pickle.load(f)
    
for key in info.keys():
    # 统计num_points_in_gt分布情况
    num=[]
    for i in range(len(info[key])):
        num.append(info[key][i]['num_points_in_gt'])
    #num_points_in_gt=info[key]['num_points_in_gt']
    num=np.array(num)
    # 统计num_points_in_gt分布情况
    print(key," points num<100:",num[num<100].shape[0])
    print(key," points num<200:",num[num<200].shape[0])
    print(key," points num<300:",num[num<300].shape[0])
    print(key," points num<400:",num[num<400].shape[0])
    print(key," points num<500:",num[num<500].shape[0])
    print(key," points num<600:",num[num<600].shape[0])
    print(key," points num<700:",num[num<700].shape[0])
    print(key," totol len:",len(num))