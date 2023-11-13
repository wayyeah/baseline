from viewer.viewer import Viewer
from pathlib import Path
# from pypcd import pypcd
import open3d as o3d
import os
import numpy as np
import cv2

def read_image(path):
    im=cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return im

import chardet
def get_label(seq, frame):
    path = r"C:\XMU_ultra\label_ego\%s\%s.json"% (seq,frame)
    assert os.path.exists(path), 'label file does not exist: '+path
    with open(path, 'r', encoding='UTF-8') as f:
        label = json.load(f)
        # label = chardet.detect(label)
    label = label['items']
    boxes = np.zeros((len(label), 7))
    classes = np.zeros((len(label), 1), dtype=str)
    for i, item in enumerate(label):
        boxes_xyz = np.zeros((3,), dtype=np.float32)
        boxes_xyz[0] = item['position']['x']
        boxes_xyz[1] = item['position']['y']
        boxes_xyz[2] = item['position']['z']
        boxes_lwh = np.zeros((3,), dtype=np.float32)
        boxes_lwh[0] = item['dimension']['x']
        boxes_lwh[1] = item['dimension']['y']
        boxes_lwh[2] = item['dimension']['z']
        boxes[i, :3] = boxes_xyz
        boxes[i, 3:6] = boxes_lwh
        boxes[i, 6] = item['rotation']['z']

        classes[i] = item['categoryId']
    return boxes, classes


def get_lidar(root_path, idx, sensor, num_features=4):
        seq = idx['seq']
        frame = idx['frame']
        
        lidar_file_list = os.listdir(os.path.join(root_path, 'seq'+seq, sensor))
        lidar_file = os.path.join(root_path, 'seq'+seq, sensor, lidar_file_list[int(frame)])
        assert Path(lidar_file).exists(), "lidar file %s not exists."%lidar_file
        # read pcd
        lidar = pypcd.PointCloud.from_path(lidar_file)
        # 转成numpy array
        # print(lidar.pc_data.dtype)
        # print(lidar.pc_data.shape, type(lidar.pc_data))

        assert num_features==4, 'only support xyzi currently'
        
        pc_x = lidar.pc_data['x']
        pc_y = lidar.pc_data['y']
        pc_z = lidar.pc_data['z']
        pc_i = lidar.pc_data['intensity']
        # print(pc_x.shape, pc_y.shape, pc_z.shape, pc_i.shape)
        # exit()
        lidar = np.stack([pc_x, pc_y, pc_z, pc_i], axis=1)
        # print(lidar.shape)

        # filter nan
        lidar = lidar[~np.isnan(lidar).any(axis=1)]

        # lidar = np.array(lidar.pc_data[:num_features], dtype=np.float32)
        # lidar = lidar[:, :num_features]
        # this line means far more to improve

        # # use o3d to read pcd instead
        # lidar2 = o3d.io.read_point_cloud(lidar_file)
        # lidar2 = np.asarray(lidar2.points)
        # lidar2.tofile('lidar2.bin')


        # # save to bin in cur path
        # lidar.tofile('lidar.bin')
        # exit()
        

        return lidar

def XMU_viewer(path):
    # root = r"I:\processing\Jun07\xianganxiaoqu_youyongguang-18-27-03"
    # root = r"I:\processing\Jun07\xianganxiaoqu_huanjingyushengtaixueyuan-18-47-57"
    # root= r"I:\processing\Jun07\huandaogandao_guanyinshan-20-14-45_night"
    # root = r"I:\processing\Jun07\xiangandaqiao-15-06-56"
    root = path if path else r"C:\XMU_ultra\20_xiangansuidao-20-02-44_03"
    

    num_frames = 200
    sensor_list = ['ouster', 'hesai', 'robosense', 'gpal']
    camera_list = ['left', 'front', 'right']

    points_dict = {}
    image_dict = {}
    for sensor in sensor_list:
        sensor_paths = os.listdir(root + '\\' + sensor)[: num_frames]
        sensor_points = []
        for i in range(num_frames):
            points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i])
            trans_mat = np.loadtxt('.\\transformation_matrix_%s.txt'%sensor)
            points = points.transform(trans_mat)
            sensor_points.append(np.asarray(points.points))           
        points_dict.update({
            sensor: sensor_points
        })
    
    for camera in camera_list:
        # camera_paths = os.listdir(root + '\\images\\' + camera)[: num_frames]
        camera_path = os.listdir(root + '\\camera_' + camera)
        camera_images = []
        for i in range(num_frames):
            # image = read_image(root + '\\images\\' + camera + '\\' + camera_paths[i])
            image = read_image(root + '\\camera_' + camera + '\\' + camera_path[i])
            camera_images.append(image)
        image_dict.update({
            camera: camera_images
        })

    vi = Viewer(box_type='OpenPCDet')

    for i in range(num_frames):
        label, _ = get_label(root.split("\\")[-1], str(i).zfill(4))
        print(i)
        for sensor in sensor_list:
            if sensor == 'ouster':
                color = 'gray'
            elif sensor == 'hesai':
                color = 'blue'
            elif sensor == 'robosense':
                color = 'green'
            elif sensor == 'gpal':
                color = 'red'
            
            if sensor == 'gpal':
                vi.add_points(points_dict[sensor][i], color=color, radius=6, scatter_filed=points_dict[sensor][i][:,2], alpha=0.2)
            elif sensor == 'hesai':
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=0.7)
            elif sensor == 'robosense':
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=0.7)
            else:
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=1.3)

                # vi.add_points(points_dict[sensor][i], color=color, radius=4, scatter_filed=points_dict[sensor][i][:,2], alpha=0.2)
            vi.add_3D_boxes(label)
        for camera in camera_list:
            # vi.add_image(image_dict[camera][i])
            cv2.namedWindow(camera, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(camera, 640, 480)
            cv2.imshow(camera, image_dict[camera][i])
        
        # vi.show_2D(box_color = (0,0, 255))
        # cv2.imshow('im',vi.image)

        cv2.waitKey(10)
        vi.show_3D()


def XMU_viewer_single():
    # root = r"I:\processing\Jun07\xianganxiaoqu_youyongguang-18-27-03"
    # root = r"I:\processing\Jun07\xianganxiaoqu_huanjingyushengtaixueyuan-18-47-57"
    # root= r"I:\processing\Jun07\huandaogandao_guanyinshan-20-14-45_night"
    # root = r"I:\processing\Jun07\xiangandaqiao-15-06-56"
    # root = r"C:\XMU_ultra\20_xiangansuidao-20-02-44_03"
    root = r"C:\XMU_ultra\07_xiangandaqiao-15-06-56_07"


    num_frames = 200
    sensor_list = ['ouster', 'hesai', 'robosense', 'gpal']
    points_dict = {}
    for sensor in sensor_list:
        sensor_paths = os.listdir(root + '\\' + sensor)[144: num_frames+144]
        print(sensor_paths)
        sensor_points = []
        for i in range(num_frames):
            points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i])
            trans_mat = np.loadtxt('.\\transformation_matrix_%s.txt'%sensor)
            points = points.transform(trans_mat)
            sensor_points.append(np.asarray(points.points))           
        points_dict.update({
            sensor: sensor_points
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
                color = 'purple'
            elif sensor == 'gpal':
                color = 'red'
            
            if sensor == 'gpal':
                vi.add_points(points_dict[sensor][i], color=color, radius=6, scatter_filed=points_dict[sensor][i][:,2], alpha=0.2)
                cv2.waitKey(10)
                vi.show_3D()
            elif sensor == 'hesai':
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=0.7)
                cv2.waitKey(10)
                vi.show_3D()

            elif sensor == 'robosense':
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=0.7)
                cv2.waitKey(10)
                vi.show_3D()

            else:
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=1.3)
                cv2.waitKey(10)
                vi.show_3D()

        # vi.show_2D(box_color = (0,0, 255))
        # cv2.imshow('im',vi.image)


def XMU_viewer_image():
    root = r"I:\processing\Jun07\xianganxiaoqu_youyongguang-18-27-03"
    # root = r"I:\processing\Jun07\xianganxiaoqu_huanjingyushengtaixueyuan-18-47-57"
    # root= r"I:\processing\Jun07\huandaogandao_guanyinshan-20-14-45_night"
    # root = r"I:\processing\Jun07\xiangandaqiao-15-06-56"


    num_frames = 20
    sensor_list = ['ouster', 'hesai', 'robosense', 'gpal']
    # camera_list = ['left', 'front', 'right']
    camera_list = ['front']

    points_dict = {}
    image_dict = {}
    for sensor in sensor_list:
        sensor_paths = os.listdir(root + '\\' + sensor)[:num_frames]
        sensor_points = []
        for i in range(num_frames):
            points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i])
            trans_mat = np.loadtxt('.\\transformation_matrix_%s.txt'%sensor)
            points = points.transform(trans_mat)
            sensor_points.append(np.asarray(points.points))           
        points_dict.update({
            sensor: sensor_points
        })
    
    for camera in camera_list:
        camera_paths = os.listdir(root + '\\images\\' + camera)[: num_frames]
        camera_images = []
        for i in range(num_frames):
            image = read_image(root + '\\images\\' + camera + '\\' + camera_paths[i])
            camera_images.append(image)
        image_dict.update({
            camera: camera_images 
        })

    vi = Viewer()

    for i in range(num_frames):
        print(i)
        for sensor in sensor_list:
            if sensor == 'ouster':
                color = 'gray'
            elif sensor == 'hesai':
                color = 'blue'
            elif sensor == 'robosense':
                color = 'green'
            elif sensor == 'gpal':
                color = 'red'
            
            if sensor == 'gpal':
                vi.add_points(points_dict[sensor][i], color=color, radius=6, scatter_filed=points_dict[sensor][i][:,2], alpha=0.2)
            elif sensor == 'hesai':
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=0.7)
            elif sensor == 'robosense':
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=0.7)
            else:
                vi.add_points(points_dict[sensor][i], color=color, radius=4, alpha=1.3)

                # vi.add_points(points_dict[sensor][i], color=color, radius=4, scatter_filed=points_dict[sensor][i][:,2], alpha=0.2)
        
        for camera in camera_list:
            vi.add_image(image_dict[camera][i])
        vi.show_2D(box_color = (0,0, 255))
    

        # for camera in camera_list:
        #     # vi.add_image(image_dict[camera][i])
        #     cv2.namedWindow(camera, cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow(camera, 640, 480)
        #     cv2.imshow(camera, image_dict[camera][i])
        # # vi.show_2D(box_color = (0,0, 255))
        # # cv2.imshow('im',vi.image)
        # cv2.waitKey(10)

        vi.show_3D()


def XMU_viewer_check_ouster():
    # root = r"I:\processing\Jun07\xianganxiaoqu_xueshengsushe-18-29-01"
    root = r"I:\processing\Jun07\maxiang_xiangdonglu_tianyuegongguan_xiangangongyeyuan-17-23-53"
    num_frames = 1000
    sensor_list = ['ouster']
    sensor_paths = os.listdir(root + '\\' + sensor_list[0])[0:num_frames]
    vi = Viewer()
    for i in range(num_frames):
        points = o3d.io.read_point_cloud(root + '\\' + sensor_list[0] + '\\' + sensor_paths[i])
        vi.add_points(np.asarray(points.points), color='gray', radius=4, alpha=1.3)
        vi.show_3D()


def XMU_viewer_done(path = None):
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
    # root = r"C:\XMU_ultra\07_xiangandaqiao-15-06-56_07"
    root = r"C:\XMU_ultra\16_maxiang_minandadao_maxaingxiaofangdadui_caicuokou-17-47-19_02"
    if path :
        root = path

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
        for i in range(num_frames):
            points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i])
            trans_mat = np.loadtxt('.\\transformation_matrix_%s.txt'%sensor)
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
                color = 'purple'
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

def XMU_viewer_done_with_label(path = None):
    # rr = r"I:\processing\Jun07\_done"
    # seq_lists = os.listdir(rr)
    # for seq in seq_lists:
    #     print(seq)
    # exit()

    # root = r"I:\processing\Jun07\_done\01_xiangandaqiao-15-06-56_01"
    # root = r"C:\XMU_ultra\03_xiangandaqiao-15-06-56_03"
    # root = r"I:\processing\Jun07\_done\48_xiangan_411xiandao_lulixiaoxue_dunhoucunkou-17-59-30_01"
    root = r"C:\XMU_ultra\27_xianganxiaoqu_huanjingyushengtaixueyuan-18-47-57_01"
    # root = r"C:\XMU_ultra\33_xiangan_411xiandao_dunhoucunkou_huangguoqingjiatingnonchang-18-04-41_01"
    # root = r"C:\XMU_ultra\20_xiangansuidao-20-02-44_03"
    root = r"C:\XMU_ultra\07_xiangandaqiao-15-06-56_07"
    if path :
        root = path

    num_frames = 20
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
        for i in range(num_frames):
            # points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i])
            points = get_lidar(root_path=root,  idx={'seq':root.split('\\')[-1], 'frame':str(i).zfill(4)}, sensor=sensor)

            trans_mat = np.loadtxt('.\\transformation_matrix_%s.txt'%sensor)
            # points = points.transform(trans_mat)
            points = np.dot(points, trans_mat)
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

    vi = Viewer(box_type='OpenPCDet')

    for i in range(num_frames):
        print(i)
        # print(lab)
        label, _ = get_label(root.split('\\')[-1], str(i).zfill(4))
        vi.add_3D_boxes(label)
        for sensor in sensor_list:
            if sensor == 'ouster':
                color = (0,64,169)
            elif sensor == 'hesai':
                color = (139,137,137)
            elif sensor == 'robosense':
                color = 'purple'
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
    
# from pypcd import pypcd
# def convert_bin_to_ascii_pcd():
#     root = r"C:\XMU_ultra\07_xiangandaqiao-15-06-56_07"
#     frame = 0
#     sensor = 'ouster'
#     sensor_paths = os.listdir(root + '\\' + sensor)[frame: frame + 1]
#     points = pypcd.PointCloud.from_path(root + '\\' + sensor + '\\' + sensor_paths[0])
#     points.save_pcd('sample_os_ascii.pcd', compression='ascii')


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
def project_to_camera_front():
    # root = r"C:\XMU_ultra\07_xiangandaqiao-15-06-56_07"
    root = r"C:\XMU_ultra\35_xiangan_411xiandao_dunhoucunkou_huangguoqingjiatingnonchang-18-04-41_03"
    # load camera intrinsic and extrinsic
    with open("ex_trinsics_front_static.json", 'r') as f:
        extrinsic = json.load(f)
    K= np.array(extrinsic['K'])
    # dist = np.array(extrinsic['dist'])
    dist = np.array([0,0,0,0,0])
    extrinsic = np.array(extrinsic['extrinsic'])

    num_frames = 200
    start_frame = 0
    assert start_frame + num_frames <=200 , 'frame number is not enough'
    sensor_list = ['ouster', 'hesai', 'robosense', 'gpal']

    # load data
    image_list = []
    points_dict = {}

    # set color
    color_ouster = (0,64,169)
    color_hesai = (139,137,137)
    color_robosense = (128,0,128)
    color_gpal = (255,0,0)

    for i in range(75, 95):
        points_all = []
        for sensor in sensor_list:
            trans_mat = np.loadtxt('.\\transformation_matrix_%s.txt'%sensor)
            sensor_paths = os.listdir(root + '\\' + sensor)[start_frame: start_frame + num_frames]
            points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i], remove_nan_points=True)
            # points = points.voxel_down_sample(voxel_size=0.01)
            points = points.transform(trans_mat)
            # add color
            if sensor == 'ouster':
                rgb = np.array([color_ouster for _ in range(len(points.points))])
            elif sensor == 'hesai':
                rgb = np.array([color_hesai for _ in range(len(points.points))])
            elif sensor == 'robosense':
                rgb = np.array([color_robosense for _ in range(len(points.points))])
            elif sensor == 'gpal':
                rgb = np.array([color_gpal for _ in range(len(points.points))])
            # concatenate points and rgb
            points = np.concatenate((np.asarray(points.points), rgb), axis=1)
            points_all.append(points)
        # stack all points
        points_u = np.vstack(points_all)
        # transform to camera coordinate
        points_u[:, :3] = world_to_camera(points_u[:, :3], extrinsic)
        points_u_2d = camera_to_pixel(points_u[:, :3], K, dist)
        img = cv2.imread(root + '\\camera_front\\' + os.listdir(root + '\\camera_front')[start_frame + i])
        img = draw_pc2image(img, points_u_2d, points_u[:, 3:])
        image_list.append(img)
        # write image to file
        # cv2.imwrite('./proj_07/image_%d.png'%i, img)
        write_path = './proj_35'
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        cv2.imwrite('./proj_35/image_%d.png'%i, img)
        print('done %d'%i)

    # # show image
    # for i in range(num_frames):
    #     print(i)
    #     cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow('image', 1920, 1080)
    #     cv2.imshow('image', image_list[i])
        

    #     cv2.waitKey(100000)

    # for sensor in sensor_list:
    #     # assert len(os.listdir(root + '\\' + sensor)) == num_frames, 'sensor %s has %d frames'%(sensor, len(os.listdir(root + '\\' + sensor)))
    #     sensor_paths = os.listdir(root + '\\' + sensor)[start_frame: start_frame + num_frames]
    #     sensor_points = []
    #     for i in range(num_frames):
    #         points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i])
    #         trans_mat = np.loadtxt('.\\transformation_matrix_%s.txt'%sensor)
    #         points = points.transform(trans_mat)
    #         points[:, :3] = world_to_camera(points[:, :3], extrinsic)

    #         sensor_points.append(np.asarray(points.points))           
    #     points_dict.update({
    #         sensor: sensor_points
    #     })
    
def project_to_camera_front_ouster():
    root = r"C:\XMU_ultra\07_xiangandaqiao-15-06-56_07"

    # load camera intrinsic and extrinsic
    with open("ex_trinsics_front_static_ouster.json", 'r') as f:
        extrinsic = json.load(f)
    K= np.array(extrinsic['K'])
    # dist = np.array(extrinsic['dist'])
    dist = np.array([0,0,0,0,0])
    extrinsic = np.array(extrinsic['extrinsic'])

    num_frames = 200
    start_frame = 0
    assert start_frame + num_frames <=200 , 'frame number is not enough'
    sensor_list = ['ouster', 'hesai', 'robosense', 'gpal']

    # load data
    image_list = []
    points_dict = {}

    # set color
    color_ouster = (0,64,169)
    color_hesai = (139,137,137)
    color_robosense = (128,0,128)
    color_gpal = (255,0,0)

    for i in range(num_frames):
        points_all = []
        for sensor in sensor_list:
            if sensor != 'ouster':
                trans_mat = np.loadtxt('.\\transformation_matrix_%s_ouster.txt'%sensor)
            sensor_paths = os.listdir(root + '\\' + sensor)[start_frame: start_frame + num_frames]
            points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i], remove_nan_points=True)
            points = points.voxel_down_sample(voxel_size=0.01)

            if sensor != 'ouster':
                points = points.transform(trans_mat)
            # add color
            if sensor == 'ouster':
                rgb = np.array([color_ouster for _ in range(len(points.points))])
            elif sensor == 'hesai':
                rgb = np.array([color_hesai for _ in range(len(points.points))])
            elif sensor == 'robosense':
                rgb = np.array([color_robosense for _ in range(len(points.points))])
            elif sensor == 'gpal':
                rgb = np.array([color_gpal for _ in range(len(points.points))])
            # concatenate points and rgb
            points = np.concatenate((np.asarray(points.points), rgb), axis=1)
            points_all.append(points)
        # stack all points
        points_u = np.vstack(points_all)
        # get rid of points backward
        points_u = points_u[points_u[:, 0] > 0]
        # transform to camera coordinate
        points_u[:, :3] = world_to_camera(points_u[:, :3], extrinsic)
        points_u_2d = camera_to_pixel(points_u[:, :3], K, dist)
        img = cv2.imread(root + '\\camera_front\\' + os.listdir(root + '\\camera_front')[start_frame + i])
        img = draw_pc2image(img, points_u_2d, points_u[:, 3:])
        image_list.append(img)
        # write image to file
        cv2.imwrite('./proj_07/image_%d.png'%i, img)

    # show image
    for i in range(num_frames):
        print(i)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1920, 1080)
        cv2.imshow('image', image_list[i])
        cv2.waitKey(100000)


def project_to_camera_ouster():
    root = r"C:\XMU_ultra\07_xiangandaqiao-15-06-56_07"

    camera_list = ['left', 'front', 'right']

    camera_dict = {}
    for camera in camera_list:
        # make directory to store images
        if not os.path.exists('./proj_07/%s'%camera):
            os.makedirs('./proj_07/%s'%camera)

        # load camera intrinsic and extrinsic
        with open("ex_trinsics_%s_static_ouster.json"%camera, 'r') as f:
            extrinsic = json.load(f)
        K= np.array(extrinsic['K'])
        # dist = np.array(extrinsic['dist'])
        # dist = np.array([0,0,0,0,0])
        dist = np.array(extrinsic['dist'])
        extrinsic = np.array(extrinsic['extrinsic'])

        num_frames = 200
        start_frame = 0
        assert start_frame + num_frames <=200 , 'frame number is not enough'
        sensor_list = ['ouster', 'hesai', 'robosense', 'gpal']

        # load data
        image_list = []

        # set color
        color_ouster = (0,64,169)
        color_hesai = (139,137,137)
        color_robosense = (128,0,128)
        color_gpal = (255,0,0)

        for i in range(num_frames):
            points_all = []
            for sensor in sensor_list:
                if sensor != 'ouster':
                    trans_mat = np.loadtxt('.\\transformation_matrix_%s_ouster.txt'%sensor)
                sensor_paths = os.listdir(root + '\\' + sensor)[start_frame: start_frame + num_frames]
                points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i], remove_nan_points=True)
                points = points.voxel_down_sample(voxel_size=0.01)

                if sensor != 'ouster':
                    points = points.transform(trans_mat)
                # add color
                if sensor == 'ouster':
                    rgb = np.array([color_ouster for _ in range(len(points.points))])
                elif sensor == 'hesai':
                    rgb = np.array([color_hesai for _ in range(len(points.points))])
                elif sensor == 'robosense':
                    rgb = np.array([color_robosense for _ in range(len(points.points))])
                elif sensor == 'gpal':
                    rgb = np.array([color_gpal for _ in range(len(points.points))])
                # concatenate points and rgb
                points = np.concatenate((np.asarray(points.points), rgb), axis=1)
                points_all.append(points)
            # stack all points
            points_u = np.vstack(points_all)
            # get rid of points backward
            points_u = points_u[points_u[:, 0] > 0]
            # transform to camera coordinate
            points_u[:, :3] = world_to_camera(points_u[:, :3], extrinsic)
            points_u_2d = camera_to_pixel(points_u[:, :3], K, dist)
            img = cv2.imread(root + '\\camera_%s\\'%camera + os.listdir(root + '\\camera_%s'%camera)[start_frame + i])
            img = draw_pc2image(img, points_u_2d, points_u[:, 3:])
            image_list.append(img)
            # write image to file
            cv2.imwrite('./proj_07/%s/image_%d.png'%(camera, i), img)
            print("done %s: %d"%(camera,i))
        camera_dict.update({
            camera: image_list
        })

    # # show image
    # for camera in camera_list:
    #     for i in range(num_frames):
    #         print(i)
    #         cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #         cv2.resizeWindow('image', 1920, 1080)
    #         cv2.imshow('image', image_list[i])
    #         cv2.waitKey(1000)

import pickle
def view_pred_gt(seq = None, frame = None, sensor = None):
    root = "C:\\XMU_ultra\\"
    for cur_seq in os.listdir(root):
        if cur_seq[:2] == seq:
            seq_path = cur_seq
            break


    # plot
    vi = Viewer(box_type='OpenPCDet')
    for frame, frame_path in enumerate(os.listdir(root + seq_path + '\\' + sensor)):
        print(frame)
        point_path = os.listdir(root + seq_path + '\\' + sensor)[int(frame)]
        path = root + seq_path + '\\' + sensor + '\\' + point_path
        points = o3d.io.read_point_cloud(path)
        trans_mat = np.loadtxt('.\\additions\\calib_to_ego\\transformation_matrix_%s_ego.txt'%sensor)
        points = points.transform(trans_mat)
        points = np.asarray(points.points)

        # get label
        label, _ = get_label(seq_path, str(frame).zfill(4))
        # get pred
        # predictions = pickle.load(open('predictions\\result_os_50.pkl', 'rb'))
        # predictions = pickle.load(open('predictions\\result_pointrcnn_Oct16.pkl', 'rb'))
        predictions = pickle.load(open('predictions\\result_os_hs_st3d++_8.pkl', 'rb'))
        for i in range(len(predictions)):
            # print(predictions[i]['frame_id']['seq'], seq, predictions[i]['frame_id']['frame_id'], frame)
            # exit()
            # print(predictions[:2], len(predictions), type(predictions))
            # print(predictions[i]['seq'], len(predictions[i]['seq']), type(predictions[i]['seq']))

            # exit()
            # print(predictions[i]['frame_id']['seq'], type(predictions[i]['frame_id']['seq']))
            # print(seq, type(seq))
            # print(predictions[i]['frame_id']['frame_id'], type(predictions[i]['frame_id']['frame_id']))
            # print(frame, type(frame))

            # print(type(seq), type(frame))
            # if predictions[i]['frame_id']['seq'] == seq and predictions[i]['frame_id']['frame_id'] == frame:
            if predictions[i]['frame_id'][:2] == seq and int(predictions[i]['frame_id'][2:]) == frame:
                pred = predictions[i]['boxes_lidar']
                pred_info = predictions[i]['score']
                break
        
        vi.add_3D_boxes(label, color='red')
        vi.add_3D_boxes(pred, color='blue', box_info=pred_info)
        vi.add_points(points, color=(0,64,169), radius=4, alpha=0.7)
        vi.show_3D()

def view_a_bin():
    path = 'predictions\\lidar.bin'
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]
    path2 = 'predictions\\lidar2.bin'
    points2 = np.fromfile(path2, dtype=np.float32).reshape(-1, 3)
    points2 = points2[:, :3]
    vi = Viewer()
    vi.add_points(points, color=(0,64,169), radius=4, alpha=0.7)
    vi.add_points(points2, color='red', radius=4, alpha=0.7)
    vi.show_3D()

def view_data_dict():
    vi = Viewer()

    # path = 'predictions\\lidar.bin'
    # points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    # points = points[:, :3]
    # vi.add_points(points, color='purple', radius=4, alpha=0.7)


    path = 'predictions\\data_dict.pkl'
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    points = data_dict['points']

    # save points
    np.savetxt('predictions\\lidar_Sep01.bin', points)

    # points_ori = data_dict['points_new']
    label = data_dict['gt_boxes']
    points = points[:, :3]

    vi.add_points(points, color=(0,64,169), radius=4, alpha=0.7)
    # vi.add_points(points_ori[:, :3], color='red', radius=4, alpha=0.7) 
    vi.add_3D_boxes(label, color='red')
    # vi.add_3D_boxes(data_dict['gt_boxes_ori'], color='blue')
    vi.show_3D()


def label_mapping():
    pass


# 高度为2.39开始转
def get_calib_to_ego():
    calib_robosense_path = r'additions\transformation_matrix_robosense.txt'
    calib_hesai_path = r'additions\transformation_matrix_hesai.txt'
    calib_ouster_path = r'additions\transformation_matrix_ouster.txt'

    calib_rs_ego_path = r'additions\calib_to_ego\transformation_matrix_robosense_ego.txt'
    calib_hs_ego_path = r'additions\calib_to_ego\transformation_matrix_hesai_ego.txt'
    calib_os_ego_path = r'additions\calib_to_ego\transformation_matrix_ouster_ego.txt'

    calib_box_ego_path = r'additions\calib_to_ego\transformation_matrix_box_ego.txt'

    calib_ouster_ego = np.array(
        [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 2.3900],
        [0, 0, 0, 1]]
    )
    np.savetxt(calib_os_ego_path, calib_ouster_ego)
    # ouster对应于ego的逆矩阵
    calib_ouster = np.loadtxt(calib_ouster_path)
    calib_ouster_neg = np.linalg.inv(calib_ouster)
    calib_box_ego = np.matmul(calib_ouster_ego, calib_ouster_neg)
    np.savetxt(calib_box_ego_path, calib_box_ego)
    
    calib_hesai = np.loadtxt(calib_hesai_path)
    # 先全部转到ouster坐标系下，然后再转到ego坐标系下
    calib_hesai_ego = np.matmul(calib_ouster_ego , np.matmul(calib_ouster_neg, calib_hesai))
    np.savetxt(calib_hs_ego_path, calib_hesai_ego)
    
    calib_robosense = np.loadtxt(calib_robosense_path)
    calib_robosense_ego = np.matmul(calib_ouster_ego , np.matmul(calib_ouster_neg, calib_robosense))
    np.savetxt(calib_rs_ego_path, calib_robosense_ego)

    print('done')

def view_in_ego():
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
    root = r"C:\XMU_ultra\20_xiangansuidao-20-02-44_03"
    # root = r"C:\XMU_ultra\07_xiangandaqiao-15-06-56_07"

    num_frames = 200
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
        trans_mat = np.loadtxt('.\\additions\\calib_to_ego\\transformation_matrix_%s_ego.txt'%sensor)
        # trans_mat_os = np.loadtxt('.\\transformation_matrix_ouster.txt')
        # trans_mat = np.matmul(np.linalg.inv(trans_mat_os),trans_mat )
        for i in range(num_frames):
            points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i])            
            points = points.transform(trans_mat)
            sensor_points.append(np.asarray(points.points)) 
            if sensor == 'gpal':
                print('shape of gpal: ', np.asarray(points.points).shape)          

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


def XMU_viewer_done_with_label_ego(path = None):
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
    # root = r"C:\XMU_ultra\07_xiangandaqiao-15-06-56_07"
    # root = r"C:\XMU_ultra\26_xianganxiaoqu_yiqicaochang-18-18-48_02"
    root = r"C:\XMU_ultra\16_maxiang_minandadao_maxaingxiaofangdadui_caicuokou-17-47-19_02"
    if path :
        root = path

    num_frames = 50
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
        for i in range(num_frames):
            points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i])
            trans_mat = np.loadtxt('.\\additions\\calib_to_ego\\transformation_matrix_%s_ego.txt'%sensor)
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

    vi = Viewer(box_type='OpenPCDet')

    for i in range(num_frames):
        print(i)
        # print(lab)
        label, _ = get_label(root.split('\\')[-1], str(i).zfill(4))
        vi.add_3D_boxes(label)
        for sensor in sensor_list:
            if sensor == 'ouster':
                color = (0,64,169)
            elif sensor == 'hesai':
                color = (139,137,137)
            elif sensor == 'robosense':
                color = 'purple'
            elif sensor == 'gpal':
                color = 'red'
            
            if sensor == 'gpal':
                # vi.add_points(points_dict[sensor][i], color=color, radius=6, scatter_filed=points_dict[sensor][i][:,2], alpha=0.2)
                vi.add_points(points_dict[sensor][i], color=color, radius=6, alpha=0.2)
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
    root = r"C:\XMU_ultra\35_xiangan_411xiandao_dunhoucunkou_huangguoqingjiatingnonchang-18-04-41_03"
    # load camera intrinsic and extrinsic
    with open("ex_trinsics_front_static.json", 'r') as f:
        extrinsic = json.load(f)
    K= np.array(extrinsic['K'])
    # dist = np.array(extrinsic['dist'])
    dist = np.array([0,0,0,0,0])
    extrinsic = np.array(extrinsic['extrinsic'])

    num_frames = 200
    start_frame = 0
    assert start_frame + num_frames <=200 , 'frame number is not enough'
    sensor_list = ['ouster', 'hesai', 'robosense', 'gpal']

    # load data
    image_list = []
    points_dict = {}

    # set color
    color_ouster = (0,64,169)
    color_hesai = (139,137,137)
    color_robosense = (128,0,128)
    color_gpal = (255,0,0)

    for i in range(75, 95):
        # draw for each sensor
        for sensor in sensor_list:
            trans_mat = np.loadtxt('.\\transformation_matrix_%s.txt'%sensor)
            sensor_paths = os.listdir(root + '\\' + sensor)[start_frame: start_frame + num_frames]
            points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i], remove_nan_points=True)
            # points = points.voxel_down_sample(voxel_size=0.01)
            points = points.transform(trans_mat)
            points_u = np.asarray(points.points)
            points_u[:, :3] = world_to_camera(points_u[:, :3], extrinsic)
            points_u_2d = camera_to_pixel(points_u[:, :3], K, dist)
            img = cv2.imread(root + '\\camera_front\\' + os.listdir(root + '\\camera_front')[start_frame + i])
            img = draw_pc2image(img, points_u_2d, points_u[:, 3:])
            write_path = './proj_35'
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            cv2.imwrite('./proj_35/image_%d_%s.png'%(i, sensor), img)
            print('done %d for sensor %s'%(i, sensor))

        exit()

        points_all = []
        for sensor in sensor_list:
            trans_mat = np.loadtxt('.\\transformation_matrix_%s.txt'%sensor)
            sensor_paths = os.listdir(root + '\\' + sensor)[start_frame: start_frame + num_frames]
            points = o3d.io.read_point_cloud(root + '\\' + sensor + '\\' + sensor_paths[i], remove_nan_points=True)
            # points = points.voxel_down_sample(voxel_size=0.01)
            points = points.transform(trans_mat)
            # add color
            if sensor == 'ouster':
                rgb = np.array([color_ouster for _ in range(len(points.points))])
            elif sensor == 'hesai':
                rgb = np.array([color_hesai for _ in range(len(points.points))])
            elif sensor == 'robosense':
                rgb = np.array([color_robosense for _ in range(len(points.points))])
            elif sensor == 'gpal':
                rgb = np.array([color_gpal for _ in range(len(points.points))])
            # concatenate points and rgb
            points = np.concatenate((np.asarray(points.points), rgb), axis=1)
            points_all.append(points)
        # stack all points
        points_u = np.vstack(points_all)
        # transform to camera coordinate
        points_u[:, :3] = world_to_camera(points_u[:, :3], extrinsic)
        points_u_2d = camera_to_pixel(points_u[:, :3], K, dist)
        img = cv2.imread(root + '\\camera_front\\' + os.listdir(root + '\\camera_front')[start_frame + i])
        img = draw_pc2image(img, points_u_2d, points_u[:, 3:])
        image_list.append(img)
        # write image to file
        # cv2.imwrite('./proj_07/image_%d.png'%i, img)
        write_path = './proj_35'
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        cv2.imwrite('./proj_35/image_%d.png'%i, img)
        print('done %d'%i)    



if __name__ == '__main__':
    # XMU_viewer()
    # XMU_viewer_single()
    # XMU_viewer_image()
    # XMU_viewer_check_ouster()
    # XMU_viewer_done()

    # mat_convert_into_ouster()

    # view_in_ouster()
    # convert_bin_to_ascii_pcd()
    # project_to_camera_front()
    # project_to_camera_front_ouster()
    # project_to_camera_ouster()
    
    # root = "C:\\XMU_ultra\\"
    # path = [root + i for i in os.listdir(root) if os.path.isdir(root + i)]
    # for i in range(4,50):
    #     print('viewing ' + path[i] + ' with label........')
        # XMU_viewer_done_with_label(path[i])

    # view_pred_gt(seq = '43', frame = 0, sensor = 'ouster')
    
    # view_a_bin()
    # view_data_dict()

    # get_calib_to_ego()
    # view_in_ego()
    # XMU_viewer_done_with_label_ego() 

    # get_gpal_info()

    scene_visulization()