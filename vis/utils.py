





import os
import subprocess
import math
import numpy as np

import cv2
import matplotlib.pyplot as plt
# install with : https://github.com/dimatura/pypcd/issues/7
# from pypcd import pypcd
import open3d as o3d

#taken from VIBE
def video_to_images(vid_file, img_folder=None, return_info=False):
    """ 
    Convert a video to images by ffmpeg
        if vid_file is None, save images to /tmp folder
    Returns:
        img_folder(, image nums, img shape)
    """
    if img_folder is None:
            img_folder = os.path.join('/tmp', os.path.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    # subprocess.call(command)
    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(os.path.join(img_folder, '000001.png')).shape
    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder

def images_to_video(img_folder, output_vid_file):
    """
    Convert images to video by ffmpeg
        images should match '%06d.png'
    """
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


# from pypcd import pypcd
# def read_pcd_with_intensity(pcd_file):

#     # verify exsitencee
#     if not os.path.exists(pcd_file):
#         raise FileNotFoundError(f'File {pcd_file} not found')

#     # points3d_pypcd = pypcd.PointCloud.from_path(pcd_file)

#     points3d = o3d.io.read_point_cloud(pcd_file)

#     if not "static" in pcd_file:
#         # transform
#         print("transforming...")
#         trans_matrix = np.loadtxt("ex_param/transformation_matrix_total_%s_static.txt" % pcd_file.split("/")[-1].split(".")[0])
#         points3d = points3d.transform(trans_matrix)


#     # intensity = np.asarray(points3d.colors)[:, 0]
#     intensity = pypcd.PointCloud.from_path(pcd_file).pc_data['intensity']
#     pc = np.concatenate((np.asarray(points3d.points), intensity[:, None]), axis=1)

#     # filter nan
#     pc = pc[~np.isnan(pc).any(axis=1)]

#     return pc


def read_pcd_with_color(pcd_file, rgb_type='depth'):
    """
    Read pcd file
        return columns: x, y, z, rgb
    Refer to:
        https://github.com/climbingdaily/LidarHumanScene/blob/master/tools/tool_func.py
    """

    # verify exsitence
    if not os.path.exists(pcd_file):
        raise FileNotFoundError(f'File {pcd_file} not found')
    
    points3d = o3d.io.read_point_cloud(pcd_file, remove_nan_points=True)

    points3d = points3d.voxel_down_sample(voxel_size=0.01)

    pc = np.asarray(points3d.points)
    if rgb_type == 'normal':
        points3d.estimate_normals()
        rgb = (np.array([0.5,0.5,0.5]) + np.array(points3d.normals) / 2) * 255
    elif rgb_type == 'depth':
        depth = np.sqrt(np.sum(pc*pc, axis=1))
        depth = depth / (np.sum(depth)/depth.shape[0]) * 255
        depth = np.array([
                155 * np.log2(i/100) / np.log2(864) + 200 if i > 200 else i \
                for i in depth
            ])
        rgb = plt.get_cmap('hsv')(depth/255)[:, :3]
    else: # may raise Error
        rgb = (np.array([0.5,0.5,0.5]) + np.array(points3d.colors) / 2) * 255
    pc = np.concatenate((pc, rgb), axis=1)
    return pc


def world_to_camera(X, extrinsic_matrix):
    n = X.shape[0]
    X = np.concatenate((X, np.ones((n, 1))), axis=-1).T
    X = np.dot(extrinsic_matrix, X).T
    return X[..., :3]

def camera_to_pixel(X, intrinsics, distortion_coefficients):
    # focal length
    f = [intrinsics[0][0], intrinsics[1][1]]
    # center principal point
    c = [intrinsics[0][2], intrinsics[1][2]]
    # k = np.array([distortion_coefficients[0],
    #              distortion_coefficients[1], distortion_coefficients[4]])
    # p = np.array([distortion_coefficients[2], distortion_coefficients[3]])

    k = np.array([distortion_coefficients[0],
                 distortion_coefficients[1], distortion_coefficients[2]])
    p = np.array([distortion_coefficients[3], distortion_coefficients[4]])

    XX = X[..., :2] / (X[..., 2:])
    # XX = pd.to_numeric(XX, errors='coere')
    r2 = np.sum(XX[..., :2]**2, axis=-1, keepdims=True)
    radial = 1 + np.sum(k * np.concatenate((r2, r2**2, r2**3),
                        axis=-1), axis=-1, keepdims=True)

    tan = 2 * np.sum(p * XX[..., ::-1], axis=-1, keepdims=True)
    XXX = XX * (radial + tan) + r2 * p[..., ::-1]
    return f * XXX + c

def draw_pc2image(image, sence_pc, colors, sensor):
    if sensor == 'gpal':
        point_size = 6
        alpha = 0.6
    else :
        point_size = 3
        alpha = 0.4
    for i, (x, y) in enumerate(sence_pc):
        rgb = colors[i]
        x = int(math.floor(x))
        y = int(math.floor(y))
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            continue
        # cv2.circle(image, (x, y), 1, color=rgb*255, thickness=-1)

        cv2.circle(image, (x, y), point_size, color=rgb*255, thickness=-1)
    return image

