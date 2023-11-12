import numpy as np
import json
import os
from pathlib import Path


def get_label(idx):
    seq = idx['seq']
    frame = idx['frame']
    # label_file = os.path.join(self.root_path, 'label', 'seq'+seq, '%s.json'%frame.zfill(4))
    root_path = 'data/xmu'
    label_file = os.path.join(root_path, 'label_ego', 'seq'+seq, '%s.json'%frame.zfill(4))
    assert Path(label_file).exists(), "label file %s not exists."%label_file
    # read json
    with open(label_file, 'r') as f:
        label = json.load(f)
    label = label['items']
    '''
    9DoF box, class, track_id, confidence, occulusion, act
    '''
    boxes_7DoF = np.zeros((len(label), 7), dtype=np.float32)
    boxes_9DoF = np.zeros((len(label), 9), dtype=np.float32)
    # classes = np.zeros((len(label), 1), dtype=str) # 为什么这么操作不对，要用list
    classes = []
    track_ids = np.zeros((len(label), 1), dtype=np.int32)
    confidences = np.zeros((len(label), 1), dtype=np.float32)
    occulusions = np.zeros((len(label), 1), dtype=np.int32)
    acts = np.zeros((len(label), 1), dtype=str)
    # points_num = np.zeros((len(label), 1), dtype=np.int32)
    points_num = np.zeros((len(label)), dtype=np.int32)
    for i, item in enumerate(label):
        boxes_xyz = np.zeros((3, ), dtype=np.float32)
        boxes_xyz[0] = item['position']['x']
        boxes_xyz[1] = item['position']['y']
        boxes_xyz[2] = item['position']['z']
        boxes_lwh = np.zeros((3, ), dtype=np.float32)
        boxes_lwh[0] = item['dimension']['x']
        boxes_lwh[1] = item['dimension']['y']
        boxes_lwh[2] = item['dimension']['z']
        boxes_pitch_roll_yaw = np.zeros((3, ), dtype=np.float32)
        boxes_pitch_roll_yaw[0] = item['rotation']['x']
        boxes_pitch_roll_yaw[1] = item['rotation']['y']
        boxes_pitch_roll_yaw[2] = item['rotation']['z']
        
        boxes_7DoF[i,:3] = boxes_xyz
        boxes_7DoF[i,3:6] = boxes_lwh
        boxes_7DoF[i,6] = boxes_pitch_roll_yaw[2]
        
        boxes_9DoF[i,:6] = boxes_7DoF[i, :6]
        boxes_9DoF[i,6:9] = boxes_pitch_roll_yaw
        # classes[i] = item['categoryId']
        classes.append(item['categoryId'])
        track_ids[i] = item['trackId']
        if 'confidence' not in item:
            confidences[i] = -1
            print('confidence error in the %d label file %s' % (i,label_file))
        # assert 'confidence' in item, 'confidence not in label file %s' % label_file
        else:
            confidences[i] = item['confidence']
        occulusions[i] = item['occ']
        acts[i] = item['act']
        points_num[i] = item['pointsNum']
        # assert len(points_num[i]) == 1, 'pointsNum length error in label file %s' % label_file
        # points_num[i] = points_num[i][0]
        # no attribute lost 
        assert item['pointsNum']>=0 , 'attribute pointsNum lost in label file %s' % label_file
        assert item['occ'] , 'attribute occ lost in label file %s' % label_file
        assert item['act'] , 'attribute act lost in label file %s' % label_file
        assert item['categoryId'] , 'attribute categoryId lost in label file %s' % label_file
        assert item['trackId'] , 'attribute trackId lost in label file %s' % label_file
    
    # # mapping 
    # assert 'TRAINING_CATEGORIES_MAPPING' in self.dataset_cfg, 'TRAINING_CATEGORIES_MAPPING not in dataset_cfg'
    # if self.dataset_cfg.TRAINING_CATEGORIES_MAPPING:
    #     # print(classes, classes.shape) 
    #     for i in range(len(classes)):
    #         if classes[i] in self.dataset_cfg.TRAINING_CATEGORIES_MAPPING:
    #             classes[i] = self.dataset_cfg.TRAINING_CATEGORIES_MAPPING[classes[i]]
    #             # 被截断了，Pedestria，要非常注意改变numpy的str类型，长度不可变
    #             assert classes[i] in ['Car', 'Truck', 'Pedestrian', 'Cyclist'], 'class %s not in [Car, Truck, Pedestrian, Cyclist]'%classes[i]
    #         else:
    #             classes[i] = 'DontCare'
    classes = np.array(classes)
    return boxes_7DoF, boxes_9DoF, classes, track_ids, confidences, occulusions, acts, points_num

import matplotlib.pyplot as plt
def label_stat():
    label_stats = {}
    label_dist = {}
    # number of points for distribution
    points_num_dist = {}
    for seq in range(1, 51):
        for frame in range(200):
            idx = {'seq': str(seq).zfill(2), 'frame': str(frame).zfill(4)}
            boxes_7DoF, boxes_9DoF, classes, track_ids, confidences, occulusions, acts, points_num = get_label(idx)
            # label number for each class; label distribution for each class, label distance distribution regardless of class, number of points for distribution, number of points for each class
            for i in range(len(classes)):
                # del Dont kown
                if classes[i] == "Don\u2019t know":
                    continue
                if classes[i] not in label_stats:
                    label_stats[classes[i]] = 1
                else:
                    label_stats[classes[i]] += 1
                # label distance distribution
                if classes[i] not in label_dist:
                    label_dist[classes[i]] = [np.sqrt(np.sum(boxes_7DoF[i,:3]**2))]
                else:
                    label_dist[classes[i]].append(np.sqrt(np.sum(boxes_7DoF[i,:3]**2)))
                # number of points for distribution
                if classes[i] not in points_num_dist:
                    points_num_dist[classes[i]] = [points_num[i]]
                else:
                    points_num_dist[classes[i]].append(points_num[i])
    # label number for each class
    print(label_stats)
    # sort and to bin
    label_stats = sorted(label_stats.items(), key=lambda item:item[1], reverse=True)
    label_stats = np.array(label_stats)
    label_stats = label_stats[:,1].astype(np.int32)
    plt.hist(label_stats, bins=100)
    plt.title('label number for each class')
    plt.savefig('label_number_for_each_class.png')
    # label distance distribution
    for key in label_dist:
        plt.hist(label_dist[key], bins=100)
        plt.title(key)
        plt.savefig('label_dist_%s.png'%key)
    # number of points for distribution
    for key in points_num_dist:
        plt.hist(points_num_dist[key], bins=100)
        plt.title(key)
        plt.savefig('points_num_dist_%s.png'%key)
    # label distribution for each class
    print(label_dist)
    # number of points for each class
    print(points_num_dist)
    
    
    

def label_stat_v2():
    class_mapping = {
        'Car': 'Car',
        'Truck': 'Truck',
        'Ped_adult': 'Adult_Ped',
        'Ped_children': 'Child_Ped',
        'Bus': 'Bus',
        'Van': 'Van',
        'Semi-Trailer towing vehicle': 'SemiTrler',
        'MotorCyc': 'MotorCyc',
        'ByCyc': 'ByCyc',
        'Tricycle': 'TriCyc',
        'Animal': 'Animal',
        'Barrier': 'Barrier',
        'Special Vehicles': 'Spcl_Veh',
    }
    # for overall label distribution
    label_distance = {}
    label_number = {}
    label_points = {}

    for seq in range(1, 51):
        for frame in range(200):
            idx = {'seq': str(seq).zfill(2), 'frame': str(frame).zfill(4)}
            boxes_7DoF, boxes_9DoF, classes, track_ids, confidences, occulusions, acts, points_num = get_label(idx)
            # label number for each class; label distribution for each class, label distance distribution regardless of class, number of points for distribution, number of points for each class
            for i in range(len(classes)):
                # del Dont kown
                if classes[i] == "Don\u2019t know":
                    continue
                # mapping
                assert classes[i] in class_mapping, 'class %s not in class_mapping'%classes[i]
                classes[i] = class_mapping[classes[i]]

                # label number for each class
                if classes[i] not in label_number:
                    label_number[classes[i]] = 1
                else:
                    label_number[classes[i]] += 1
                # label distance distribution
                if classes[i] not in label_distance:
                    label_distance[classes[i]] = [np.sqrt(np.sum(boxes_7DoF[i,:3]**2))]
                else:
                    label_distance[classes[i]].append(np.sqrt(np.sum(boxes_7DoF[i,:3]**2)))
                # number of points for distribution
                if classes[i] not in label_points:
                    label_points[classes[i]] = [points_num[i]]
                else:
                    label_points[classes[i]].append(points_num[i])

    # for class specific label distribution
    label_number_class = {}
    # sort with label number
    label_number = sorted(label_number.items(), key=lambda item:item[1], reverse=True)
    label_number = np.array(label_number)
    print(label_number)
    # save label_number as xlsx
    # import pandas as pd
    # df = pd.DataFrame(label_number)
    # df.to_excel('label_number_for_each_class.xlsx')
    # save label_number as txt
    np.savetxt('label_number_for_each_class.txt', label_number, fmt='%s')
    # save to npy
    np.save('label_number_for_each_class.npy', label_number)
    # draw as histogram
    plt.figure()
    plt.bar(label_number[:,0], label_number[:,1].astype(np.int32))
    plt.xticks(rotation=45)
    # plt.title('label number for each class')
    plt.tight_layout()
    plt.savefig('label_number_for_each_class.png')
    
    # save pkl
    import pickle
    with open('label_number_for_each_class.pkl', 'wb') as f:    
        pickle.dump(label_number, f)
    with open('label_points_for_each_class.pkl', 'wb') as f:
        pickle.dump(label_points, f)

    # for class agnostic label distribution
    label_distance_all = []
    label_points_all = []
    for key in label_distance:
        label_distance_all += label_distance[key]
    for key in label_points:
        label_points_all += label_points[key]
    # draw as histogram
    plt.figure()
    plt.hist(label_distance_all, bins=100)
    # plt.title('label distance distribution')
    plt.tight_layout()
    plt.savefig('label_distance_distribution.png')
    # save to npy
    np.save('label_distance_distribution.npy', label_distance_all)
    plt.figure()
    plt.hist(label_points_all, bins=100)
    plt.title('label points distribution')
    plt.tight_layout()
    plt.savefig('label_points_distribution.png')
    # save to npy
    np.save('label_points_distribution.npy', label_points_all)


    # # for sensor specific label distribution
    # print(type(label_points))
    # print(label_points['Car'])



if __name__ == '__main__':
    # label_stat()
    label_stat_v2()