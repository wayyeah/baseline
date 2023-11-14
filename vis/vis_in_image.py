
from pathlib import Path
from pypcd import pypcd
import os
import numpy as np
import cv2

def read_image(path):
    im=cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return im


def get_lidar(idx, sensor, num_features=4):
    seq = idx['seq']
    frame = idx['frame']
    root_path = '/home/xmu/datasets/XMechanismUnmanned'
    if sensor == 'camera':
        raise NotImplementedError
    
    lidar_file_list = os.listdir(os.path.join(root_path, 'seq'+seq, sensor))
    lidar_file_list.sort() #一定要sort，不然顺序不对
    lidar_file = os.path.join(root_path, 'seq'+seq, sensor, lidar_file_list[int(frame)])
    assert Path(lidar_file).exists(), "lidar file %s not exists."%lidar_file
    assert num_features==4, 'only support xyzi currently'
    if sensor=='gpal':
        lidar_file = os.path.join("/home/xmu/datasets/XMechanismUnmanned_gpalcsv", 'seq'+seq, sensor, lidar_file_list[int(frame)])
        lidar_file=lidar_file.replace('.pcd', '.npy')
        assert Path(lidar_file).exists(), "lidar file %s not exists."%lidar_file
        lidar=np.load(lidar_file) # (N, 6)
        lidar =lidar[:,:4] 
    else:
        lidar = pypcd.PointCloud.from_path(lidar_file)
        pc_x = lidar.pc_data['x']
        pc_y = lidar.pc_data['y']
        pc_z = lidar.pc_data['z']
        pc_i = lidar.pc_data['intensity']
        lidar = np.stack([pc_x, pc_y, pc_z, pc_i], axis=1)
    
    to_ego = True
    if to_ego:
        # transform points to ego vehicle coord with calib using matrix multiplication
        calib_file = os.path.join(root_path, 'calib_to_ego', 'transformation_matrix_%s_ego.txt'%sensor)
        assert Path(calib_file).exists(), "calib file %s not exists."%calib_file
        # read calib
        calib = np.loadtxt(calib_file, dtype=np.float32)
        lidar_new = np.concatenate((lidar[:, :3], np.ones((lidar.shape[0], 1))), axis=1)
        lidar_new1 = np.dot(calib, lidar_new.T).T # 这样才是对的！！！
        lidar = np.concatenate((lidar_new1[:, :3], lidar[:, 3:]), axis=1)
    lidar = lidar[lidar[:, 0]>=0]
    # for density

    # filter nan
    lidar = lidar[~np.isnan(lidar).any(axis=1)]
    
    
    #裁切点云
    radius_range = [0, 150]  # 半径范围
    angle_range = [-np.pi / 3, np.pi / 3]  # 60度到负60度，转换为弧度
    cropped_points = crop_sector(lidar, radius_range, angle_range)
    lidar=cropped_points  
    lidar = lidar[:, :4]

    # transform back to world coord
    if to_ego:
        # transform points to ego vehicle coord with calib using matrix multiplication
        calib_file = os.path.join(root_path, 'calib_to_ego', 'transformation_matrix_%s_ego.txt'%sensor)
        assert Path(calib_file).exists(), "calib file %s not exists."%calib_file
        # read calib
        calib = np.loadtxt(calib_file, dtype=np.float32)
        calib = np.linalg.inv(calib)
        lidar_new = np.concatenate((lidar[:, :3], np.ones((lidar.shape[0], 1))), axis=1)
        lidar_new1 = np.dot(calib, lidar_new.T).T # 这样才是对的！！！
        lidar = np.concatenate((lidar_new1[:, :3], lidar[:, 3:]), axis=1)

    # save to pcd
    # lidar = lidar[:, :4]
    # lidar = lidar.astype(np.float32)
    # pc = pypcd.PointCloud.from_array(lidar)
    # pc.save_pcd('test.pcd', compression='binary_compressed')

    

    return lidar

def get_lidar_o(idx, sensor, num_features=4):
    seq = idx['seq']
    frame = idx['frame']
    root_path = '/home/xmu/datasets/XMechanismUnmanned'
    if sensor == 'camera':
        raise NotImplementedError
    
    lidar_file_list = os.listdir(os.path.join(root_path, 'seq'+seq, sensor))
    lidar_file_list.sort() #一定要sort，不然顺序不对
    lidar_file = os.path.join(root_path, 'seq'+seq, sensor, lidar_file_list[int(frame)])
    assert Path(lidar_file).exists(), "lidar file %s not exists."%lidar_file
    assert num_features==4, 'only support xyzi currently'
    if sensor=='gpal':
        lidar_file = os.path.join("/home/xmu/datasets/XMechanismUnmanned_gpalcsv", 'seq'+seq, sensor, lidar_file_list[int(frame)])
        lidar_file=lidar_file.replace('.pcd', '.npy')
        assert Path(lidar_file).exists(), "lidar file %s not exists."%lidar_file
        lidar=np.load(lidar_file) # (N, 6)
        lidar =lidar[:,:4] 
    else:
        lidar = pypcd.PointCloud.from_path(lidar_file)
        pc_x = lidar.pc_data['x']
        pc_y = lidar.pc_data['y']
        pc_z = lidar.pc_data['z']
        pc_i = lidar.pc_data['intensity']
        lidar = np.stack([pc_x, pc_y, pc_z, pc_i], axis=1)
    
    to_ego = True
    if to_ego:
        # transform points to ego vehicle coord with calib using matrix multiplication
        calib_file = os.path.join(root_path, 'calib_to_ego', 'transformation_matrix_%s_ego.txt'%sensor)
        assert Path(calib_file).exists(), "calib file %s not exists."%calib_file
        # read calib
        calib = np.loadtxt(calib_file, dtype=np.float32)
        lidar_new = np.concatenate((lidar[:, :3], np.ones((lidar.shape[0], 1))), axis=1)
        lidar_new1 = np.dot(calib, lidar_new.T).T # 这样才是对的！！！
        lidar = np.concatenate((lidar_new1[:, :3], lidar[:, 3:]), axis=1)
    lidar = lidar[lidar[:, 0]>=0]
    # for density

    # filter nan
    lidar = lidar[~np.isnan(lidar).any(axis=1)]
    
    
    #裁切点云
    radius_range = [0, 150]  # 半径范围
    angle_range = [-np.pi / 3, np.pi / 3]  # 60度到负60度，转换为弧度
    cropped_points = crop_sector(lidar, radius_range, angle_range)
    lidar=cropped_points  
    lidar = lidar[:, :4]

    return lidar



def view_in_ouster():
    # rr = r"I:\processing\Jun07\_done"
    # seq_lists = os.listdir(rr)
    # for seq in seq_lists:
    #     print(seq)
    # exit()

    # root = r"I:\processing\Jun07\_done\01_xiangandaqiao-15-06-56_01"
    # root = r"C:\XMU_ultra\03_xiangandaqiao-15-06-56_03"
    # root = r"I:\processing\Jun07\_done\48_xiangan_411xiandao_lulixiaoxue_dunhoucunkou-17-59-30_01"
    # root = r"C:\XMU_ultra\27_xianganxiaoqu_huanjingyushengtaixueyuan-18-47-57_01"
    # root = r"C:\XMU_ultra\33_xiangan_411xiandao_dunhoucunkou_huangguoqingjiatingnonchang-18-04-41_01"
    # root = r"C:\XMU_ultra\20_xiangansuidao-20-02-44_03"
    root = r"C:\XMU_ultra\07_xiangandaqiao-15-06-56_07"

    num_frames = 10
    start_frame = 0
    assert start_frame + num_frames <=200 , 'frame number is not enough'
    sensor_list = ['ouster', 'hesai', 'robosense', 'gpal']
    camera_list = ['left', 'front', 'right']

    points_dict = {}
    image_dict = {}
    for sensor in sensor_list:
        # assert len(os.listdir(root + '\\' + sensor)) == num_frames, 'sensor %s has %d frames'%(sensor, len(os.listdir(root + '\\' + sensor)))
        sensor_paths = os.listdir(root + '\\' + sensor)[start_frame: start_frame + num_frames]
        sensor_points = []
        if not sensor == 'ouster':
            trans_mat = np.loadtxt('.\\transformation_matrix_%s_ouster.txt'%sensor)
        # trans_mat_os = np.loadtxt('.\\transformation_matrix_ouster.txt')
        # trans_mat = np.matmul(np.linalg.inv(trans_mat_os),trans_mat )
        for i in range(num_frames):
            points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i])            
            if not sensor == 'ouster':
                points = points.transform(trans_mat)
            sensor_points.append(np.asarray(points.points))           
        points_dict.update({
            sensor: sensor_points
        })
    
    for camera in camera_list:
        camera_paths = os.listdir(root + '\\camera_' + camera)[start_frame: start_frame + num_frames]
        # assert len(os.listdir(root + '\\camera_' + camera)) == num_frames, 'camera %s has %d frames'%(camera, len(os.listdir(root + '\\camera_' + camera)))
        camera_images = []
        for i in range(num_frames):
            image = read_image(root + '\\camera_' + camera + '\\' + camera_paths[i])
            camera_images.append(image)
        image_dict.update({
            camera: camera_images
        })

    vi = Viewer()

    for i in range(num_frames):
        print(i)
        for sensor in sensor_list:
            if sensor == 'ouster':
                color = (0,64,169)
            elif sensor == 'hesai':
                color = (139,137,137)
            elif sensor == 'robosense':
                color = (128,0,128)
            elif sensor == 'gpal':
                color = 'red'
            
            if sensor == 'gpal':
                vi.add_points(points_dict[sensor][i], color=color, radius=6, scatter_filed=points_dict[sensor][i][:,2], alpha=0.2)
            elif sensor == 'hesai':
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=0.7)
            elif sensor == 'robosense':
                # vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=0.7)
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=0.5)
            else:
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=0.7)

                # vi.add_points(points_dict[sensor][i], color=color, radius=4, scatter_filed=points_dict[sensor][i][:,2], alpha=0.2)
        for camera in camera_list:
            # vi.add_image(image_dict[camera][i])
            cv2.namedWindow(camera, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(camera, 640, 480)
            cv2.imshow(camera, image_dict[camera][i])
        

        cv2.waitKey(10)
        vi.show_3D()

def mat_convert_into_ouster():
    # transform all into ouster
    trans_mat_ouster = np.loadtxt('.\\transformation_matrix_ouster.txt')

    sensor_list = ['hesai', 'robosense', 'gpal']
    for sensor in sensor_list:
        trans_mat = np.matmul(np.linalg.inv(trans_mat_ouster), np.loadtxt('.\\transformation_matrix_%s.txt'%sensor))
        np.savetxt(".\\transformation_matrix_%s_ouster.txt"%sensor, trans_mat)
        print("done converting %s"%sensor)
    
    camera_list = ['left', 'front', 'right']
    for camera in camera_list:
        ex_trinsics = json.load(open("ex_trinsics_%s_static.json"%camera))
        extrinsic = np.array(ex_trinsics['extrinsic'])
        extrinsic = np.matmul(extrinsic, trans_mat_ouster)
        ex_trinsics['extrinsic'] = extrinsic.tolist()
        json.dump(ex_trinsics, open("ex_trinsics_%s_static_ouster.json"%camera, 'w'))
        print("done converting %s"%camera)


def read_points_with_sensor_info(pcd_file = None, sensor = None):
    if not pcd_file or not sensor:
        raise ValueError('pcd_file and sensor should be specified')
    
    # read pcd file
    points = o3d.io.read_point_cloud(pcd_file)
    # read transformation matrix


import json
from utils import world_to_camera
from utils import camera_to_pixel
from utils import draw_pc2image

def crop_sector(points, radius_range, angle_range):
    # 计算极坐标系下的半径和角度
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    
    # 将角度转换到 [-π, π) 范围
    theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
    
    # 应用裁剪条件
    mask = (r >= radius_range[0]) & (r <= radius_range[1]) & (theta >= angle_range[0]) & (theta <= angle_range[1])
    cropped_points = points[mask]
    
    return cropped_points

from matplotlib import pyplot as plt
def get_gpal_info():
    root = r"C:\XMU_ultra"
    seq_list = os.listdir(root)
    seq_list = [i for i in seq_list if i[:1] >= '0' and i[:1] <= '9']
    assert len(seq_list) == 50, 'seq number is not 50'
    sensor_points = []
    for i in seq_list:
        sensor = 'gpal'
        sensor_paths = os.listdir(root + '\\' + i + '\\' + sensor)

        trans_mat = np.loadtxt('.\\additions\\calib_to_ego\\transformation_matrix_%s_ego.txt'%sensor)
        for j in sensor_paths:
            points = o3d.io.read_point_cloud(root + '\\' + i + '\\' + sensor + '\\' + j)
            # to ego
            points = points.transform(trans_mat)
            
            points  = np.asarray(points.points)
            #裁切点云
            radius_range = [0, 150]  # 半径范围
            angle_range = [-np.pi / 3, np.pi / 3]  # 60度到负60度，转换为弧度
            cropped_points = crop_sector(points, radius_range, angle_range)
            points=cropped_points

            sensor_points.append(points)
    # count points per frame
    points_num = []
    for i in sensor_points:
        points_num.append(i.shape[0])
    # save to npy
    points_num = np.array(points_num)
    print(points_num.shape)
    np.save('gpal_points_num.npy', points_num)
    # draw histogram
    plt.hist(points_num, bins=100)
    plt.xlabel('points number')
    plt.ylabel('frame number')
    plt.savefig('gpal_points_num.png')
    plt.clf()

    # real points
    points = np.concatenate(sensor_points)
    print(points.shape)
    # down sample to 1 million
    np.random.shuffle(points)
    points = points[:1000000]
    # save to npy
    np.save('gpal_points.npy', points)

    # distance distrribution
    distance = []
    for i in sensor_points:
        distance.append(np.sqrt(i[:, 0] ** 2 + i[:, 1] ** 2))
    distance = np.concatenate(distance)
    print(distance.shape)
    # randomly down_sample_to_1million
    np.random.shuffle(distance)
    distance = distance[:1000000]
    # save to npy
    np.save('gpal_distance.npy', distance)
    # draw histogram
    plt.hist(distance, bins=100)
    plt.xlabel('distance')
    plt.ylabel('points number')
    plt.savefig('gpal_distance.png')
    plt.clf()
    


def scene_visulization():
    root = '/home/xmu/datasets/XMechanismUnmanned'
    # load camera intrinsic and extrinsic
    with open("./proj_trans_to_server/ex_trinsics_front_static.json", 'r') as f:
        extrinsic = json.load(f)
    K= np.array(extrinsic['K'])
    # dist = np.array(extrinsic['dist'])
    dist = np.array([0,0,0,0,0])
    extrinsic = np.array(extrinsic['extrinsic'])

    sensor_list = ['ouster', 'hesai', 'robosense', 'gpal']
    # load data
    image_list = []
    points_dict = {}

    # set color
    color_ouster = (0,64,169)
    color_hesai = (139,137,137)
    color_robosense = (128,0,128)
    color_gpal = (255,0,0)

    seq = '35'
    for frame in range(50, 200):
        # draw for each sensor
        for sensor in sensor_list:
            trans_mat = np.loadtxt('./proj_trans_to_server/transformation_matrix_%s.txt'%sensor)
            idx = {'seq': seq, 'frame': str(frame).zfill(4)}
            points = get_lidar(idx, sensor)
            # transform with trans_mat
            points_new = np.concatenate((points[:, :3], np.ones((points.shape[0], 1))), axis=1)
            points_new1 = np.dot(trans_mat, points_new.T).T # 这样才是对的！！！
            points = np.concatenate((points_new1[:, :3], points[:, 3:]), axis=1)

            points_u = np.zeros((points.shape[0], 6))
            points_u[:, :4] = points[:, :4]
            points_u[:,:3] = world_to_camera(points_u[:, :3], extrinsic)
            points_u_2d = camera_to_pixel(points_u[:, :3], K, dist)
            # intensity to rgb
            points_u[:, 3:] = intensity_to_rgb(points_u[:, 3])
            # print(root + '/seq35/camera_front/')
            img_path_list = os.listdir(root + '/seq35/camera_front/')
            img_path_list.sort()
            # print(img_path_list)
            img_path = root + '/seq35/camera_front/' + img_path_list[frame]
            img = cv2.imread(img_path)
            img = draw_pc2image(img, points_u_2d, points_u[:, 3:],sensor=sensor)
            write_path = './proj_35'
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            cv2.imwrite('./proj_35/image_%d_%s.png'%(frame, sensor), img)
            print('done %d for sensor %s'%(frame, sensor))

        # points_all = []
        # for sensor in sensor_list:
        #     trans_mat = np.loadtxt('.\\transformation_matrix_%s.txt'%sensor)
        #     sensor_paths = os.listdir(root + '\\' + sensor)[start_frame: start_frame + num_frames]
        #     points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i], remove_nan_points=True)
        #     # points = points.voxel_down_sample(voxel_size=0.01)
        #     points = points.transform(trans_mat)
        #     # add color
        #     if sensor == 'ouster':
        #         rgb = np.array([color_ouster for _ in range(len(points.points))])
        #     elif sensor == 'hesai':
        #         rgb = np.array([color_hesai for _ in range(len(points.points))])
        #     elif sensor == 'robosense':
        #         rgb = np.array([color_robosense for _ in range(len(points.points))])
        #     elif sensor == 'gpal':
        #         rgb = np.array([color_gpal for _ in range(len(points.points))])
        #     # concatenate points and rgb
        #     points = np.concatenate((np.asarray(points.points), rgb), axis=1)
        #     points_all.append(points)
        # # stack all points
        # points_u = np.vstack(points_all)
        # # transform to camera coordinate
        # points_u[:, :3] = world_to_camera(points_u[:, :3], extrinsic)
        # points_u_2d = camera_to_pixel(points_u[:, :3], K, dist)
        # img = cv2.imread(root + '\\camera_front\\' + os.listdir(root + '\\camera_front')[start_frame + i])
        # img = draw_pc2image(img, points_u_2d, points_u[:, 3:])
        # image_list.append(img)
        # # write image to file
        # # cv2.imwrite('./proj_07/image_%d.png'%i, img)
        # write_path = './proj_35'
        # if not os.path.exists(write_path):
        #     os.makedirs(write_path)
        # cv2.imwrite('./proj_35/image_%d.png'%i, img)
        # print('done %d'%i)    


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def intensity_to_rgb(intensity):
    # 归一化强度值到 [0, 1]
    normalized_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    # 将归一化后的强度值映射到 HSV 色彩空间
    hue = normalized_intensity * 0.8  # 0.8 是 HSV 色彩空间中红到绿的范围
    saturation = 1.0
    value = 1.0
    # 将 HSV 转换为 RGB
    rgb_color = hsv_to_rgb(np.stack((hue, saturation * np.ones_like(hue), value * np.ones_like(hue)), axis=-1))
    
    return rgb_color


def get_vis_sample_to_npy():
    sensors = ['ouster', 'robosense', 'hesai', 'gpal']
    for sensor in sensors:
        points = get_lidar_o({'seq': '35', 'frame': '0081'}, sensor)
        save_path = r'./vis_sample/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + sensor + '.npy', points)
        print('done %s'%sensor)

if __name__ == '__main__':
    # scene_visulization()
    get_vis_sample_to_npy()