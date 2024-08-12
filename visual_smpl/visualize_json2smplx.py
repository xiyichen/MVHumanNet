''' 
1. Download SMPLX_NEUTRAL.pkl(about 518 mb) from https://github.com/vchoutas/smplx#downloading-the-model). 
2. Put SMPLX_NEUTRAL.pkl to the path of "\smpl_\smplx"
'''

import os
import cv2
import sys
# os.environ["PYOPENGL_PLATFORM"] = 'egl'

from mytools.smplmodel import  load_model
from os.path import join
import numpy as np
import trimesh

import natsort
import glob
import argparse
import shutil

from mytools.camera_utils import read_camera_mvhumannet
from mytools.writer import FileWriter

import matplotlib.pyplot as plt
import pickle

import pdb

def center_crop_with_keypoint(img, crop_width, crop_height, keypoint_x, keypoint_y, K, resized_cropped_size):
    """
    Center the image crop around a specified keypoint.

    Parameters:
    img (numpy.ndarray): The original image.
    crop_width (int): The width of the crop.
    crop_height (int): The height of the crop.
    keypoint_x (int): The x-coordinate of the keypoint.
    keypoint_y (int): The y-coordinate of the keypoint.

    Returns:
    numpy.ndarray: The cropped image.
    int: The new x-offset of the top-left corner.
    int: The new y-offset of the top-left corner.
    """
    # Get the dimensions of the image
    height, width, _ = img.shape

    # Calculate the starting points to center the keypoint in the cropped image
    start_x = max(0, keypoint_x - crop_width // 2)
    start_y = max(0, keypoint_y - crop_height // 2)

    # Ensure the crop does not go beyond the image boundaries
    end_x = min(width, start_x + crop_width)
    end_y = min(height, start_y + crop_height)

    # Adjust the start points if the crop would go out of bounds
    start_x = max(0, end_x - crop_width)
    start_y = max(0, end_y - crop_height)

    # Crop the image
    cropped_img = img[start_y:end_y, start_x:end_x]
    
    resized_cropped_img = cv2.resize(cropped_img[:,:,::-1],[resized_cropped_size, resized_cropped_size], interpolation=cv2.INTER_AREA)
    
    K[0, 2] -= start_x
    K[1, 2] -= start_y
    
    
    K[0][0] *= (resized_cropped_size / crop_width)
    K[1][1] *= (resized_cropped_size / crop_height)
    K[0][2] *= (resized_cropped_size / crop_width)
    K[1][2] *= (resized_cropped_size / crop_height)

    return resized_cropped_img, K


cam_i = "CC32871A015"
img_path = f'/cluster/scratch/xiychen/100050/images_lr/{cam_i}/0005_img.jpg'
mask_path = f'/cluster/scratch/xiychen/100050/fmask_lr/{cam_i}/0005_img_fmask.png'
kpts_path = f'/cluster/scratch/xiychen/100050/openpose/{cam_i}/0005_img_keypoints.json'
with open(kpts_path) as f:
    import json
    kpts = np.array(json.load(f)['people'][0]['pose_keypoints_2d']).reshape(-1, 3) / 2
# json_filename = r'data/000075.json'
json_filename = r'/cluster/scratch/xiychen/100050/smplx/smpl/000000.json'
# intri_name = r'data/camera_intrinsics.json'
intri_name = r'/cluster/scratch/xiychen/100050/camera_intrinsics.json'
# extri_name =  r'data/camera_extrinsics.json'
extri_name = r'/cluster/scratch/xiychen/100050/camera_extrinsics.json'

# cam_i = "CC32871A059"
# camera_scale_fn = r'data/camera_scale.pkl'
camera_scale_fn = r'/cluster/scratch/xiychen/100050/camera_scale.pkl'

import cv2
import numpy as np

def add_transparent_background(img, mask):
    """
    Adds a transparent background to an image using a binary mask.

    Parameters:
    img (numpy.ndarray): The original image with shape (h, w, c).
    mask (numpy.ndarray): The binary mask with shape (h, w), values should be 0 or 1.

    Returns:
    numpy.ndarray: The image with a transparent background (h, w, 4).
    """
    # Ensure the mask is in the correct format (h, w) and values are 0 or 255
    mask = mask.astype(np.uint8) * 255

    # Convert the mask to have the same shape as the image (h, w, 1)
    alpha_channel = np.expand_dims(mask, axis=-1)

    # Combine the image with the alpha channel
    rgba_img = np.concatenate([img, alpha_channel], axis=-1)

    return rgba_img



camera_scale = pickle.load(open(camera_scale_fn, "rb"))
if camera_scale ==120 / 65:
    image_size = [4096, 3000]
else:
    image_size = [2448, 2048]

image_size[0] //= 2
image_size[1] //= 2


from mytools.reader import read_smpl



mesh = trimesh.load('/cluster/scratch/xiychen/100050/smplx/smplx_mesh/000000.obj')
vertices = np.array(mesh.vertices)
min_coords = vertices.min(axis=0)
max_coords = vertices.max(axis=0)
center = (min_coords + max_coords) / 2

param = read_smpl(json_filename)[0]
body_model = load_model(gender='neutral', model_type='smplx',model_path='smpl_')
vertices = (body_model(return_verts=True, return_tensor=False, **param) - center)
# joints = body_model(return_verts=False, return_smpl_joints=True, return_tensor=False, **param) - center
mesh = trimesh.Trimesh(vertices=vertices[0], faces=body_model.faces)
# mesh.export(outpath)
vertices = np.array(vertices[0])
faces = np.array(body_model.faces)

cameras_gt = read_camera_mvhumannet(intri_name, extri_name,camera_scale)



render_data = {}
assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
# pid = self.pid
pid ,nf = 0,0
render_data = {'vertices': vertices, 'faces': faces, 'vid': pid, 'name': 'human_'}
cameras = {'K': [], 'R':[], 'T':[]}

sub_vis = [cam_i]
# if len(sub_vis) == 0:
    # sub_vis = ["22327084"]
for key in cameras.keys():
    cameras[key] = np.stack([cameras_gt[cam][key] for cam in sub_vis])



cameras['RT'] = np.zeros((1,3,4))
cameras['RT'][:,:,:3] = cameras['R']
cameras['RT'][:,:,3] = cameras['T'].reshape(-1,1,3)
cameras['position'] = -np.transpose(cameras['RT'][:,:3,:3], (0,2,1))@cameras['RT'][:,:3,3:]

cameras['T'][0] = -cameras['R']@(cameras['position'][0] - center.reshape(3,1))
cropped_size = 1100
resized_cropped_size = 256
# pdb.set_trace()
cameras['K'][0][0][0]/=2
cameras['K'][0][0][2]/=2
cameras['K'][0][1][1]/=2
cameras['K'][0][1][2]/=2


config={}
write_smpl  = FileWriter("", config=config)

data_images = cv2.imread(img_path)
data_images = cv2.resize(data_images,[image_size[0], image_size[1]], interpolation=cv2.INTER_AREA) # h,w,c
mask = cv2.imread(mask_path)//255
pelvis = kpts[8][:2].astype(np.int32)
K = cameras['K'][0]
data_images, K = center_crop_with_keypoint(data_images, cropped_size, cropped_size, pelvis[0], pelvis[1], K.copy(), resized_cropped_size)
mask, _ = center_crop_with_keypoint(mask, cropped_size, cropped_size, pelvis[0], pelvis[1], K.copy(), resized_cropped_size)
mask = mask[:,:,0]
cameras['K'][0] = K
# cameras['K'][0][0][0] *= 2
# cameras['K'][0][1][1] *= 2
# cameras['K'][0][0][2] *= 2
# cameras['K'][0][1][2] *= 2
# cameras['T'] *= 0.5

cameras['RT'] = np.zeros((1,3,4))
cameras['RT'][:,:,:3] = cameras['R']
cameras['RT'][:,:,3] = cameras['T'].reshape(-1,1,3)
cameras['position'] = -np.transpose(cameras['RT'][:,:3,:3], (0,2,1))@cameras['RT'][:,:3,3:]

print(cameras)

render_data_input = {"0":render_data}

# NoSuchDisplayException: Cannot connect to "None"
outname_cache = (r'smplx.jpg')
smpl_img = write_smpl.vis_smpl(render_data_input, [data_images], cameras, outname_cache, add_back=False)
smpl_img_re = cv2.resize(smpl_img,[image_size[0], image_size[1]], interpolation=cv2.INTER_AREA)
smpl_img_re = cv2.resize(smpl_img,[resized_cropped_size, resized_cropped_size], interpolation=cv2.INTER_AREA)

# plt.imshow(smpl_img_re)
# plt.imshow(data_images)

# plt.imshow(data_images +smpl_img_re[:,:,:3])

smpl_img_re_mask = 1- smpl_img_re[:,:,3:4]//255
# import pdb
# pdb.set_trace()

# # smpl_img_re_mask
# plt.imshow(smpl_img_re_mask)

# import pdb
# pdb.set_trace()

data_images[mask==0] = 255
import cv2
cv2.imwrite('./whitebg.png', data_images[:,:,::-1])
img_output = (data_images*smpl_img_re_mask +smpl_img_re[:,:,:3]).astype("uint8")

# plt.imshow((data_images*smpl_img_re_mask +smpl_img_re[:,:,:3]).astype("uint8"))
name = "json2smplx.png"

# data_images_rgba = add_transparent_background(data_images, mask)
# import pdb
# pdb.set_trace()

plt.imsave(name, img_output)
# plt.imsave(name, data_images_rgba)
# from PIL import Image
# img = Image.fromarray(data_images_rgba, 'RGBA')            
# img.save(name)
# cv2.imwrite(name, data_images_rgba)
# cv2.imwrite(name, img_output)
                
        
        
        
        
        
        
        
    
