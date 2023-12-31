import numpy as np
import pickle
from pathlib import Path
from pypcd import pypcd
import os
import torch
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
root_path="data/xmu"
def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data
def get_lidar(idx, sensor, num_features=4):
    seq = idx['seq']
    frame = idx['frame']
    if sensor == 'camera':
        raise NotImplementedError
    lidar_file_list = os.listdir(os.path.join(root_path, 'seq'+seq, sensor))
    lidar_file_list.sort() #一定要sort，不然顺序不对
    lidar_file = os.path.join(root_path, 'seq'+seq, sensor, lidar_file_list[int(frame)])
    assert Path(lidar_file).exists(), "lidar file %s not exists."%lidar_file
    lidar = pypcd.PointCloud.from_path(lidar_file)
    assert num_features==4, 'only support xyzi currently'
    pc_x = lidar.pc_data['x']
    pc_y = lidar.pc_data['y']
    pc_z = lidar.pc_data['z']
    pc_i = lidar.pc_data['intensity']
    pc_ring = lidar.pc_data['ring']
    lidar = np.stack([pc_x, pc_y, pc_z, pc_i, pc_ring], axis=1)
    # print(lidar.shape)
    to_ego = True
    if to_ego:
        # transform points to ego vehicle coord with calib using matrix multiplication
        calib = get_calib(idx=idx, sensor=sensor)
        lidar_new = np.concatenate((lidar[:, :3], np.ones((lidar.shape[0], 1))), axis=1)
        lidar_new1 = np.dot(calib, lidar_new.T).T # 这样才是对的！！！
        lidar = np.concatenate((lidar_new1[:, :3], lidar[:, 3:]), axis=1)
    lidar = lidar[lidar[:, 0]>=0]
    lidar = lidar[~np.isnan(lidar).any(axis=1)]
    lidar = lidar[:, :4]
    return lidar
def get_calib(idx, sensor):
    calib_file = os.path.join(root_path, 'calib_to_ego', 'transformation_matrix_%s_ego.txt'%sensor)
    assert Path(calib_file).exists(), "calib file %s not exists."%calib_file
    calib = np.loadtxt(calib_file, dtype=np.float32)
    return calib
print("input result path:")
result_path=input()
result=read_pkl(result_path)
print("source sensor: 0:ouster,1:robosense,2:hesai")
sensors=["ouster","robosense","hesai"]
source_sensor=sensors[int(input())]
print("target sensor: 0:ouster,1:robosense,2:hesai")
target_sensor=sensors[int(input())]

split="train"
score_dic={'Car':0.9,'Truck':0.7,'Pedestrian':0.5,'Cyclist':0.5}
print("source sensor:",source_sensor," target sensor:" ,target_sensor," set:",split)

data_path='data/xmu'
dis_thresh=70.4
#print(result[0])
used_classes=["Car", "Truck","Pedestrian", "Cyclist"]
database_save_path = Path(data_path) / ('pseudo_gt_database_%s2%s' % (source_sensor,target_sensor) if split == 'train' else 'gt_database_%s2%s_%s' % (source_sensor,target_sensor, split))
db_info_save_path = Path(data_path) / ('pseudo_gt_database_info_%s2%s.pkl' % (source_sensor,target_sensor) if split == 'train' else 'pseudo_gt_database_info_%s2%s_%s.pkl' % (source_sensor,target_sensor, split))
print("save Path:",database_save_path)
print('len info when create gt_database: ', len(result))
database_save_path.mkdir(parents=True, exist_ok=True)
all_db_infos = {}
print('generation gt_database for sensor: %s2%s' % (source_sensor,target_sensor))
for i in range(len(result)):
    print('%s gt_database sample: %d/%d' % (target_sensor, i+1, len(result)))
    info = result[i]
    #print(info.keys())
    idx=(info['frame_id'])
    idx['frame']=idx['frame_id']
    points = get_lidar(idx=idx, sensor=target_sensor, num_features=4)
    annos =info
    gt_boxes_7D=annos['boxes_lidar']
    gt_names = annos['name']
    gt_scores = annos['score']
    # 初始化一个布尔数组，用于标记过滤后的元素
    mask = np.zeros(len(gt_names), dtype=bool)

    # 遍历所有类别名称，更新mask
    for name in np.unique(gt_names):
        if name in score_dic:
            # 对于每个类别，找到该类别的所有框，并检查它们的得分是否高于阈值
            class_mask = (gt_names == name) & (gt_scores > score_dic[name])
            # 更新总mask
            mask |= class_mask  # 等同于 mask = mask | class_mask

    # 应用mask过滤gt_boxes_7D和gt_names
    gt_boxes_7D = gt_boxes_7D[mask]
    gt_names = gt_names[mask]
    num_gt = gt_boxes_7D.shape[0]
    
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes_7D)
    ).numpy() # (num_gt, num_points)

    for i in range(num_gt):
        file_name = '%s_%s_%s_%d.bin' % (idx['seq'], idx['frame'], gt_names[i], i)
        file_path = database_save_path / file_name
        gt_points = points[point_indices[i, :] > 0]
        
        if gt_points.shape[0] == 0: # 滤掉没有点的gt_box
            continue
       
        dis=np.sqrt(np.sum(gt_boxes_7D[i][:3]**2,axis=0))
        if dis>dis_thresh:
            continue
        gt_points[:, :3] -= gt_boxes_7D[i][:3]
        
        with open(file_path, 'wb') as f:
            
            gt_points.tofile(f)
        #np.save("/home/xmu/projects/xmuda/yw/OpenPCDet/points.npy",points)
        
        if (used_classes is None) or (gt_names[i] in used_classes):
            db_path = str(file_path.relative_to(root_path))
            db_info = {'name': gt_names[i], 'path': db_path, 'gt_idx': i,
                        'box3d_lidar': gt_boxes_7D[i], 'num_points_in_gt': gt_points.shape[0]}
            if gt_names[i] in all_db_infos.keys():
                all_db_infos[gt_names[i]].append(db_info)
            else:
                all_db_infos[gt_names[i]] = [db_info]

for k, v in all_db_infos.items():
    print('sensor--%s database of %s : %d' % (target_sensor, k, len(v)))
with open(db_info_save_path, 'wb') as f:
    pickle.dump(all_db_infos, f)