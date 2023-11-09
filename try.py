import numpy as np
import os
from pypcd import pypcd
from pathlib import Path
def downsample_ouster_to_hesai():
    ouster_fov_range = (-22.5, 22.5)
    hesai_fov_range = (-16, 15)
    ouster_angles = np.linspace(ouster_fov_range[0], ouster_fov_range[1], 128)
    hesai_angles = np.linspace(hesai_fov_range[0], hesai_fov_range[1], 32)

    ouster_angles_keep = np.array([False for i in range(128)])

    for hesai_angle in hesai_angles:
        # Find the closest ouster angle
        closest_ouster_angle_idx = np.argmin(np.abs(ouster_angles - hesai_angle))
        # Use this index to mask the points and rings
        ouster_angles_keep[closest_ouster_angle_idx] = True
    
    lines_kept = np.array([i for i in range(128) if ouster_angles_keep[i] == True])
    return ouster_angles_keep, lines_kept

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



def get_lidar(idx, sensor, num_features=4):
    seq = idx['seq']
    frame = idx['frame']
    root_path = "data/xmu"
    
    lidar_file_list = os.listdir(os.path.join(root_path, 'seq'+seq, sensor))
    lidar_file_list.sort() #一定要sort，不然顺序不对
    lidar_file = os.path.join(root_path, 'seq'+seq, sensor, lidar_file_list[int(frame)])
    assert Path(lidar_file).exists(), "lidar file %s not exists."%lidar_file
    assert num_features==4, 'only support xyzi currently'
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
    lidar = lidar[lidar[:, 0]>0]   
       
            
    # filter nan
    lidar = lidar[~np.isnan(lidar).any(axis=1)]
    lidar = lidar[:, :4]
    
     #裁切点云
    radius_range = [0, 150]  # 半径范围
    angle_range = [-np.pi / 3, np.pi / 3]  # 60度到负60度，转换为弧度
    cropped_points = crop_sector(lidar, radius_range, angle_range)
    lidar=cropped_points
    return lidar

from matplotlib import pyplot as plt
import math
def get_and_draw_original_intensity_statistic(sensors=None):
    print('We are using original intensity.....')
    # random choise 300 frames from training set
    root  = '/home/xmu/projects/xmuda/OpenPCDet/data/xmu'
    path_training_set = '/home/xmu/projects/xmuda/OpenPCDet/data/xmu/ImageSets/train.txt'
    with open(path_training_set, 'r') as f:
        lines = f.readlines()
    # for each of the 30 training sequences, randomly select 20 frames
    seqs = [line.strip() for line in lines]
    frames = np.random.choice(200, 20, replace=False)
    
    # count the intensity for each sensor with 30 seqs
    for sensor in sensors:
        sensor_intensity = []
        for seq in seqs:
            for frame in frames:
                idx = {'seq':seq, 'frame':frame}
                lidar = get_lidar(idx, sensor)
                intensity = lidar[:,3]
                sensor_intensity.append(intensity)
        sensor_intensity = np.concatenate(sensor_intensity)
        save_path = os.path.join('intensity', sensor)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'intensity.npy'), sensor_intensity)
        # sensor, number of points, min, max, mean, std
        print('sensor: ', sensor, '\t', \
              'number_of_points: ', sensor_intensity.shape, '\t', \
              'min: ', np.min(sensor_intensity), '\t', \
              'max: ', np.max(sensor_intensity), '\t', \
              'mean: ', np.mean(sensor_intensity), '\t', \
              'std: ', np.std(sensor_intensity)
              )
        plt.hist(sensor_intensity, bins=125, cumulative=False, label=sensor)
        # save the figure for each sensor
        plt.legend()
        plt.savefig('intensity_statistic_{}.png'.format(sensor))
        plt.clf()
        print('done generating statistic of sensor {}'.format(sensor))
    
    # draw the original intensity figure for all the sensors
    for sensor in sensors:
        # get data from npy
        save_path = os.path.join('intensity', sensor, 'intensity.npy')
        assert os.path.exists(save_path), 'npy path for sensor {} do not exist!'.format(sensor)
        intensity = np.load(save_path)
        # print the percentage of points with intensity > 500
        print('sensor: ', sensor, '\t', 'percentage of points with intensity > 500: ', np.sum(intensity>500)/intensity.shape[0])
        intensity = intensity[intensity<=500]
        # draw intensity: use percentage of intensity as y axis, use intensity as x axis and draw (max - min) number of points
        plt.hist(intensity, bins=math.ceil(np.max(intensity) - np.min(intensity)), cumulative=False, label=sensor, density=True)
    plt.legend()
    plt.savefig('intensity_statistic_all.png')
    plt.clf()


from scipy import stats
def convert_and_draw_cox_box_intensity(sensors=None):
    print('We are using box-cox intensity.....')
    # '''
    intensitys = {}
    # draw the box-cox intensity figure for all the sensors
    lambda_values = {}
    for sensor in sensors:
        save_path = os.path.join('intensity', sensor, 'intensity.npy')
        assert os.path.exists(save_path), 'npy path for sensor {} do not exist!'.format(sensor)
        intensity = np.load(save_path)

        # for box-cox, we need all the intensity > 0
        intensity = intensity + 1e-6

        # conduct box-cox
        transformed_intensity, lambda_value =  stats.boxcox(intensity)
        intensitys[sensor] = transformed_intensity
        # write to file
        np.save(os.path.join('intensity', sensor, 'intensity_boxcox.npy'), transformed_intensity)
        # recalculating the mean and std
        print('sensor: ', sensor, '\t', \
                'labda_value: ', lambda_value, '\t', \
                'transformed_intensity.shape: ', transformed_intensity.shape, '\t', \
                'min: ', np.min(transformed_intensity), '\t',
                'max: ', np.max(transformed_intensity), '\t',
                'mean: ', np.mean(transformed_intensity), '\t', 
                'std: ', np.std(transformed_intensity)
            )
        
        lambda_values[sensor] = lambda_value

        # save figure
        plt.hist(intensity, bins=125, cumulative=False, label=sensor+'_box-cox', density=True)
        plt.legend()
        plt.savefig('intensity_statistic_boxcox_{}.png'.format(sensor))
        plt.clf()
    print('done generating box-cox statistic of all sensors')
    print('lambda_values: ', lambda_values)
    # '''

    # draw the box-cox intensity figure for all the sensors
    for sensor in sensors:
        # get data from npy
        save_path = os.path.join('intensity', sensor, 'intensity_boxcox.npy')
        assert os.path.exists(save_path), 'npy path for sensor {} do not exist!'.format(sensor)
        intensity = np.load(save_path)
        plt.hist(intensity, bins=125, cumulative=False, label=sensor+'_box-cox', density=True)
    plt.legend()
    plt.savefig('intensity_statistic_boxcox_all.png')
    plt.clf()


def get_and_draw_log_intensity_statistic(sensors=None):
    print('We are using log intensity.....')
    # draw the box-cox intensity figure for a single sensor
    for sensor in sensors:
        # get data from npy
        save_path = os.path.join('intensity', sensor, 'intensity.npy')
        assert os.path.exists(save_path), 'npy path for sensor {} do not exist!'.format(sensor)
        intensity = np.load(save_path)
        # log
        intensity = np.log(intensity + 1)
        # save to file
        np.save(os.path.join('intensity', sensor, 'intensity_log.npy'), intensity)
        print('sensor: ', sensor, '\t', \
                'log_intensity.shape: ', intensity.shape, '\t', \
                'min: ', np.min(intensity), '\t',
                'max: ', np.max(intensity), '\t',
                'mean: ', np.mean(intensity), '\t',
                'std: ', np.std(intensity)
            )
        # draw
        plt.hist(intensity, bins=math.ceil(np.max(intensity) - np.min(intensity)), cumulative=False, label=sensor+'_log', density=True)
        plt.legend()
        plt.savefig('intensity_statistic_log_{}.png'.format(sensor))
        plt.clf()
    # draw the box-cox intensity figure for all the sensors
    for sensor in sensors:
        # get data from npy
        save_path = os.path.join('intensity', sensor, 'intensity_log.npy')
        assert os.path.exists(save_path), 'npy path for sensor {} do not exist!'.format(sensor)
        intensity = np.load(save_path)
        # draw
        plt.hist(intensity, bins=125, cumulative=False, label=sensor+'_log', density=True)
    plt.legend()
    plt.savefig('intensity_statistic_log_all.png')
    plt.clf()


def get_and_draw_log_boxcox_intensity_statistic(sensors=None):
    print('We are using log box-cox intensity.....')
    lambda_values = {}
    intensitys = {}
    # draw the box-cox intensity figure for a single sensor
    for sensor in sensors:
        # get data from numpy 
        save_path = os.path.join('intensity', sensor, 'intensity_log.npy')
        assert os.path.exists(save_path), 'npy path for sensor {} do not exist!'.format(sensor)
        intensity = np.load(save_path)
        # box-cox
        print('sensor: ', sensor, '\t', 'before box-cox: ', intensity.shape, '\t', 'min: ', np.min(intensity), '\t', 'max: ', np.max(intensity), '\t', 'mean: ', np.mean(intensity), '\t', 'std: ', np.std(intensity))
        transformed_intensity, lambda_value =  stats.boxcox(intensity+1e-6)
        lambda_values[sensor] = lambda_value
        # save to file
        np.save(os.path.join('intensity', sensor, 'intensity_log_boxcox.npy'), transformed_intensity)
        print('sensor: ', sensor, '\t', \
                'labda_value: ', lambda_value, '\t', \
                'transformed_intensity.shape: ', transformed_intensity.shape, '\t', \
                'min: ', np.min(transformed_intensity), '\t',
                'max: ', np.max(transformed_intensity), '\t',
                'mean: ', np.mean(transformed_intensity), '\t',
                'std: ', np.std(transformed_intensity)
            )
        # draw
        plt.hist(transformed_intensity, bins=math.ceil(np.max(transformed_intensity) - np.min(transformed_intensity)), cumulative=False, label=sensor+'_log_boxcox', density=True)
        plt.legend()
        plt.savefig('intensity_statistic_log_boxcox_{}.png'.format(sensor))
        plt.clf()
    print('done generating log box-cox statistic of all sensors')
    print('lambda_values: ', lambda_values)
    # draw the box-cox intensity figure for all the sensors
    for sensor in sensors:
        save_path = os.path.join('intensity', sensor, 'intensity_log_boxcox.npy')
        assert os.path.exists(save_path), 'npy path for sensor {} do not exist!'.format(sensor)
        transformed_intensity = np.load(save_path)
        # draw
        plt.hist(transformed_intensity, bins=125, cumulative=False, label=sensor+'_log_boxcox', density=True)
    plt.legend()
    plt.savefig('intensity_statistic_log_boxcox_all.png')
    plt.clf()


def compare_one_frame_for_all(sensors=None, idx=None):
    # sensors = ['ouster', 'hesai', 'robosense']
    box_cox_lambdas = {'ouster': 0.09407519012989105, 'hesai': 0.29870034429070486, 'robosense': -0.09467222971312698}
    box_cox_lambdas_log = {'ouster': 0.5194996882001862, 'hesai': 0.592819354971455, 'robosense': 0.5432802611595038}
    # select one frame
    if idx is None:
        idx = {'seq':'01', 'frame':0}
    figure_save_dir = 'figure_intensity/seq{}_frame{}'.format(idx['seq'], idx['frame'])
    if not os.path.exists(figure_save_dir):
        os.makedirs(figure_save_dir)
    # before box-cox
    processings = ['original', 'log', 'boxcox', 'log_boxcox', 'log_boxcox_norm', 'log_boxcox_norm_pre']

    for processing in processings:
        for sensor in sensors:
            lidar = get_lidar(idx, sensor)
            intensity = lidar[:,3]
            if processing == 'log':
                intensity = np.log(intensity + 1)
            elif processing == 'boxcox':
                intensity = stats.boxcox(intensity + 1e-6, box_cox_lambdas[sensor])
            elif processing == 'log_boxcox':
                intensity = stats.boxcox(np.log(intensity + 1) + 1e-6, box_cox_lambdas_log[sensor])
            elif processing == 'log_boxcox_01norm':
                intensity = stats.boxcox(np.log(intensity + 1) + 1e-6, box_cox_lambdas_log[sensor])
                # pc_i = ((pc_i - np.mean(pc_i))/ np.std(pc_i)) * (np.max(pc_i) - np.min(pc_i)) + np.min(pc_i)
                intensity = ((intensity - np.mean(intensity))/ np.std(intensity)) * (np.max(intensity) - np.min(intensity)) + np.min(intensity)
            elif processing == 'log_boxcox_norm_pre':
                box_cox_lambdas_log_pre = {'ouster':0.529603806963828, 'hesai':0.5931315699272153, 'robosense':0.4900647247193385}
                intensity = stats.boxcox(np.log(intensity + 1) + 1e-6, box_cox_lambdas_log_pre[sensor])
                intensity = ((intensity - np.mean(intensity))/ np.std(intensity)) * (np.max(intensity) - np.min(intensity)) + np.min(intensity)
                
            # draw
            plt.hist(intensity, bins=125, cumulative=False, label=sensor+'_'+processing, density=True)
        plt.legend() 
        
        plt.savefig(os.path.join(figure_save_dir, 'intensity_statistic_{}.png'.format(processing)))
        plt.clf()
    

def all_process_with_log():
    sensors = ['ouster', 'hesai', 'robosense']
    '''
    # random choise 300 frames from training set
    root  = '/home/xmu/projects/xmuda/OpenPCDet/data/xmu'
    path_training_set = '/home/xmu/projects/xmuda/OpenPCDet/data/xmu/ImageSets/train.txt'
    with open(path_training_set, 'r') as f:
        lines = f.readlines()
    seqs = [line.strip() for line in lines]
    # print(len(lines), lines)
    # sample 10 frames from each seq
    # random sample 5 frames from each seq with 200 frames
    frames = np.random.choice(200, 10, replace=False)
    
    # count the intensity for each sensor with 30 seqs
    for sensor in sensors:
        sensor_intensity = []
        cnt_sensor = 0
        for seq in seqs:
            for frame in frames:
                idx = {'seq':seq, 'frame':frame}
                lidar = get_lidar(idx, sensor)
                intensity = lidar[:,3]
                sensor_intensity.append(intensity)
                cnt_sensor += intensity.shape[0]
        sensor_intensity = np.concatenate(sensor_intensity)
        sensor_intensity = np.log(sensor_intensity + 1)
        print(sensor, sensor_intensity.shape)
        save_path = os.path.join('intensity', sensor)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'intensity_log.npy'), sensor_intensity)

    for sensor in sensors:
        intensity = np.load(os.path.join('intensity', sensor, 'intensity_log.npy'))
        intensity = intensity + 1e-6

        # _, _ = stats.boxcox()

        # conduct box-cox
        transformed_intensity, lambda_value =  stats.boxcox(intensity)
        # write to file
        np.save(os.path.join('intensity', sensor, 'intensity_log_boxcox.npy'), transformed_intensity)
        # recalculating the mean and std
        print(sensor, 'labda_value: ', lambda_value,transformed_intensity.shape, np.mean(transformed_intensity), np.std(transformed_intensity))
    '''
    box_cox_lambda = {'ouster':0.529603806963828, 'hesai':0.5931315699272153, 'robosense':0.4900647247193385}
    # select one frame
    idx = {'seq':'01', 'frame':0}
    for sensor in sensors:
        intensity = np.load(os.path.join('intensity', sensor, 'intensity_log_boxcox.npy'))
        # normalization to [0,1] with mean==0.5
        # intensity = (intensity - np.min(intensity))/(np.max(intensity) - np.min(intensity))
        intensity = ((intensity - np.mean(intensity))/ np.std(intensity)) * (np.max(intensity) - np.min(intensity)) + np.min(intensity)
        # save figure
        plt.hist(intensity, bins=125, cumulative=False, label=sensor)
    plt.legend()
    plt.savefig('intensity_statistic_boxcox_log_01norm.png')

def count_points(sensors=None):
    pass


def get_and_draw_intensity_statistic():
    sensors = ['ouster', 'hesai', 'robosense']
    # get_and_draw_original_intensity_statistic(sensors)
    # convert_and_draw_cox_box_intensity(sensors)
    # get_and_draw_log_intensity_statistic(sensors)
    # get_and_draw_log_boxcox_intensity_statistic(sensors)
    compare_one_frame_for_all(sensors, idx={'seq':'44', 'frame':5})


import os
def rename_seq_lab():
    root_path = "data/xmu/label_ego"
    for path in os.listdir(root_path):
        os.popen('mv %s/%s %s/%s'%(root_path, path, root_path, 'seq'+path[:2]))
        print("done renaming %s"%path)


def density_statistic():
    # sensors = ['ouster', 'hesai', 'robosense', 'gpal']
    sensors = ['ouster', 'hesai', 'robosense']
    # sensors = ['ouster']
    # count the point of every frame from each sensor
    points_sensor = {}
    # more_points_seq = []
    for sensor in sensors:
        print('sensor: ', sensor)
        cnt_sensor = []
        points_sensor[sensor] = []
        for seq in range(1,51):
            for frame in range(200):
                idx = {'seq':str(seq).zfill(2), 'frame':str(frame).zfill(3)}
                lidar = get_lidar(idx, sensor)
                cnt_sensor.append(lidar.shape[0])
                # if lidar.shape[0] > 200000:
                    # more_points_seq.append(seq)
                points_sensor[sensor].append(lidar)
        cnt_sensor = np.array(cnt_sensor)
        # more_points_seq remove same seq
        # more_points_seq = list(set(more_points_seq))
        # print('more_points_seq: ', more_points_seq)

        # save lidar to npy
        points_sensor[sensor] = np.array(points_sensor[sensor])
        # concatenate all the points

        save_path = os.path.join('points', sensor)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'points_spare.npy'), points_sensor[sensor])

        points_sensor[sensor] = np.concatenate(points_sensor[sensor])

        np.save(os.path.join(save_path, 'points.npy'), points_sensor[sensor])

        # draw the histogram
        plt.hist(cnt_sensor, bins=125, cumulative=False, label=sensor, density=True)
        # save the figure for each sensor
        plt.legend()
        plt.savefig('density_statistic_{}.png'.format(sensor))
        plt.clf()

    # draw distance statistic
    for sensor in sensors:
        points_sensor[sensor] = np.load(os.path.join('points', sensor, 'points.npy'), allow_pickle=True)
        print(sensor, points_sensor[sensor].shape)
        # points_sensor[sensor] = np.reshape(points_sensor[sensor], (-1,3))
        # print(sensor, points_sensor[sensor].shape)
        # draw the histogram
        plt.hist(points_sensor[sensor][:,0], bins=125, cumulative=False, label=sensor, density=True, alpha=0.5)
    plt.legend()
    plt.savefig('density_statistic_distance_all.png')
    plt.clf()

    # for all the sensors
    for sensor in sensors:
        points_sensor[sensor] = np.load(os.path.join('points', sensor, 'points_spare.npy'), allow_pickle=True)
        # draw the histogram of number points per frame
        cnt_sensor = []
        for i in range(points_sensor[sensor].shape[0]):
            cnt_sensor.append(points_sensor[sensor][i].shape[0])
        cnt_sensor = np.array(cnt_sensor)
        plt.hist(cnt_sensor, bins=125, cumulative=False, label=sensor, density=True, alpha=0.5)
    plt.legend()
    plt.savefig('density_statistic_frame_all.png')
    plt.clf()


def density_stat():
    sensors = ['ouster']
    for sensor in sensors:
        points_sensor = np.load(os.path.join('points', sensor, 'points.npy'), allow_pickle=True)
        print(sensor, points_sensor.shape)
        # filter points with sqrt(x^2 + y^2) <1
        points_sensor = points_sensor[points_sensor[:,0]**2 + points_sensor[:,1]**2 < 1]
        print(sensor, points_sensor.shape)
    


if __name__ == "__main__":
    # ouster_angles_keep, lines_kept = downsample_ouster_to_hesai()
    # print(ouster_angles_keep)
    # print(lines_kept)

    # get_and_draw_intensity_statistic()
    # rename_seq_lab()

    density_statistic()
    # density_stat()


