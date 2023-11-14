from pcdet.datasets.once.once_eval.evaluation import get_evaluation_results
import copy
import numpy as np
import pickle
def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

gt_path='/home/xmu/projects/xmuda/baseline/data/xmu/xmu_infos_val.pkl'
result_path=None
try:
    gt_pkl=read_pkl(gt_path)
except:
    gt_path=None
try:
    det_pkl=read_pkl(result_path)
except:
    result_path=None
if gt_path is None:
    print("输入gt路径")
    gt_path=input()
if result_path is None:
    print("输入result路径")
    result_path=input()
iou_set={'Car': 0.5,'Pedestrian': 0.25,'Cyclist': 0.25,'Truck': 0.5}


class_names=['Car','Truck','Pedestrian','Cyclist']
gt_pkl=read_pkl(gt_path)
det_pkl=read_pkl(result_path)
eval_det_annos = copy.deepcopy(det_pkl)
eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
ap_result_str, ap_dict = get_evaluation_results(gt_annos=eval_gt_annos, pred_annos=eval_det_annos, classes=class_names,iou_thresholds=iou_set)

print(ap_result_str)