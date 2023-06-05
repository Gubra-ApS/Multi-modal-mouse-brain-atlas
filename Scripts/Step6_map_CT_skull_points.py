## Import libraries
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
import scipy.ndimage.interpolation as sci_int
import scipy.ndimage.morphology as sci_morph

from skimage import data, color, img_as_uint
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread, imshow
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

from skimage import segmentation
from skimage.morphology import disk, ball
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage import morphology, transform
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.restoration import (denoise_tv_chambolle)

import pandas as pd
import sys
from subprocess import call, run
import subprocess
from math import sqrt
import SimpleITK as sitk
from shutil import move
import math
#import seaborn as sns
import skimage.filters as sf


#### Import custom libraries
import sys
sys.path.insert(0, '/opt/imaging/libraries')
import importlib
import GubraImg as gi
importlib.reload(gi)

# Path input/output
folder_in = r'/CT_2_MRI/mri_brains_aligned/'
folder_out = r'/CT_2_MRI/mri_brains_aligned/'

if not os.path.exists( folder_out): os.makedirs( folder_out )

# Selected reference brain from study
reference_brain_id = r'ID008'

# Path to ccfv3 atlas
cffv3_dir = r'/ccfv3_25um'

# elastix path
elastix_path = r'/elastix'

# List of samples to include in atlas
sample_id = ['ID012','ID007','ID011','ID008','ID010','ID002','ID004','ID003','ID001','ID005','ID006','ID009']

# CT bregma/lambda
ct_path = []
for i in range(0,len(sample_id)):
    ct_path.append(os.path.join(folder_in, sample_id[i]+'_bregma1_lambda2.nii.gz'))


# REGISTRATIONS ############################################################

#### 1) Run aff + average of all to A0
fixed = folder_out + reference_brain_id + '_ccfv3_rig_matched.nii.gz'
# do registrations
for n in range(len(sample_id)):
    moving = folder_out + sample_id[n] + '_bregma1_lambda2.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.transform_vol(moving, folder_out+sample_id[n]+'_aff_A0', sample_id[n]+'_bregma1_lambda2_aff_A0', type='ano')


#### 2) Run bspline + average of all to A1
fixed = folder_out + 'A0.nii.gz'
# do registrations
for n in range(len(sample_id)):
    moving = folder_out + sample_id[n] + '_bregma1_lambda2_aff_A0.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.transform_vol(moving, folder_out+sample_id[n]+'_bspline_A1', sample_id[n]+'_bregma1_lambda2_bspline_A1', type='ano')


#### 3) Run bspline + average of all to A2
fixed = folder_out + 'A1.nii.gz'
# do registrations
for n in range(len(sample_id)):
    moving = folder_out + sample_id[n] + '_bregma1_lambda2_bspline_A1.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.transform_vol(moving, folder_out+sample_id[n]+'_bspline_A2', sample_id[n]+'_bregma1_lambda2_bspline_A2', type='ano')


#### 4) Run bspline + average of all to A3
fixed = folder_out + 'A2.nii.gz'
# do registrations
for n in range(len(sample_id)):
    moving = folder_out + sample_id[n] + '_bregma1_lambda2_bspline_A2.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.transform_vol(moving, folder_out+sample_id[n]+'_bspline_A3', sample_id[n]+'_bregma1_lambda2_bspline_A3', type='ano')


#### 5) Run bspline + average of all to A4
fixed = folder_out + 'A3.nii.gz'
# do registrations
for n in range(len(sample_id)):
    moving = folder_out + sample_id[n] + '_bregma1_lambda2_bspline_A3.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.transform_vol(moving, folder_out+sample_id[n]+'_bspline_A4', sample_id[n]+'_bregma1_lambda2_bspline_A4', type='ano')


#### 6) Run bspline + average of all to A5
fixed = folder_out + 'A4.nii.gz'
# do registrations
for n in range(len(sample_id)):
    moving = folder_out + sample_id[n] + '_bregma1_lambda2_bspline_A4.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.transform_vol(moving, folder_out+sample_id[n]+'_bspline_A5', sample_id[n]+'_bregma1_lambda2_bspline_A5', type='ano')


# AVERAGE THE BREGMA AND ALMBDA POINTS (not in symmetric volume)##################

# Average bregma and lambda points and fill in the volume
avg = gi.io.load_nifti(folder_in + 'A5.nii.gz')
avg_points = np.zeros(avg.shape)

# For bregma
z = []
y = []
x = []
for n in range(len(sample_id)):
    temp = gi.io.load_nifti(folder_in + sample_id[n] +'_bregma1_lambda2_bspline_A5.nii.gz')
    if 1 in temp:
        z.append(np.argwhere(temp==1)[0][0])
        y.append(np.argwhere(temp==1)[0][1])
        x.append(np.argwhere(temp==1)[0][2])
z_avg_bregma = np.around(np.sum(z)/len(z))
y_avg_bregma = np.around(np.sum(y)/len(y))
x_avg_bregma = np.around(np.sum(x)/len(x))
print(len(z))
avg_points[int(z_avg_bregma), int(y_avg_bregma), int(x_avg_bregma)]=1

# For Lambda
z = []
y = []
x = []
for n in range(len(sample_id)):
    temp = gi.io.load_nifti(folder_in + sample_id[n] +'_bregma1_lambda2_bspline_A5.nii.gz')
    if 2 in temp:
        z.append(np.argwhere(temp==2)[0][0])
        y.append(np.argwhere(temp==2)[0][1])
        x.append(np.argwhere(temp==2)[0][2])
z_avg_lambda = np.around(np.sum(z)/len(z))
y_avg_lambda = np.around(np.sum(y)/len(y))
x_avg_lambda = np.around(np.sum(x)/len(x))
print(len(z))
avg_points[int(z_avg_lambda), int(y_avg_lambda), int(x_avg_lambda)]=2

# Save
gi.io.save_nifti(avg_points.astype('uint8'), folder_out + 'AVG_bregma1_lambda2_bspline_A5.nii.gz')


# ROTATE BRGMA/LAMBDA AND A5 ###############################################
fixed = cffv3_dir + r'/ccfv3_template_olf_pad.nii.gz'
moving1 = folder_out + 'A5.nii.gz'
moving2 = folder_out + 'AVG_bregma1_lambda2_bspline_A5.nii.gz'
hu = gi.registration.Elastix(moving1,fixed,elastix_path,folder_out)
hu.registration('Bspline_Gubra_Aug2021_coarse.txt', 'A5_straight')
hu = gi.registration.Elastix(moving2,fixed,elastix_path,folder_out)
hu.transform_vol(moving2, folder_out+'A5_straight', 'AVG_bregma1_lambda2_bspline_A5_straight', type='ano')

moving1 = gi.io.load_nifti(folder_out + 'A5_straight.nii.gz')
moving1_rot = ndimage.rotate(moving1, angle = 3.8, axes = (0,1),reshape=False)
gi.io.save_nifti(moving1_rot, folder_out + 'A5_straight_rot.nii.gz')
moving2 = gi.io.load_nifti(folder_out + 'AVG_bregma1_lambda2_bspline_A5_straight.nii.gz')
moving2_rot = ndimage.rotate(moving2, angle = 3.8, axes = (0,1),reshape=False,order =0)
gi.io.save_nifti(moving2_rot, folder_out + 'AVG_bregma1_lambda2_bspline_A5_straight_rot.nii.gz')


# SYMMETRY AND MASKING AVERAGE ###############################################
## Average
avg = gi.io.load_nifti(folder_out + 'A5_straight_rot.nii.gz')
symm_avg = np.zeros(np.shape(avg))
symm_avg[:,:,:228] = avg[:,:,:228]
avg_flip = np.flip(avg[:,:,:227], axis = 2)
symm_avg[:,:,228:455] = avg_flip
symm_avg = symm_avg[18:315,19:634,:455]
gi.io.save_nifti(symm_avg.astype('uint8'), folder_out + 'A5_straight_rot_symm2.nii.gz')

## Mask
# Transform to Allen
fixed = cffv3_dir + r'/ccfv3_template_olf_pad.nii.gz'
moving2 = folder_out + 'tissue_mask_A5_man.nii.gz'
hu = gi.registration.Elastix(moving2,fixed,elastix_path,folder_out)
hu.transform_vol(moving2, folder_out+'A5_straight', 'tissue_mask_A5_man_straight', type='ano')

# Rotate
moving2 = gi.io.load_nifti(folder_out + 'tissue_mask_A5_man_straight.nii.gz')
moving2_rot = ndimage.rotate(moving2, angle = 3.8, axes = (0,1),reshape=False,order =0)
gi.io.save_nifti(moving2_rot, folder_out + 'tissue_mask_A5_man_straight_rot.nii.gz')

# Make symmetric
mask = gi.io.load_nifti(folder_out + 'tissue_mask_A5_man_straight_rot.nii.gz')
symm_mask = np.zeros(np.shape(mask))
symm_mask[:,:,:228] = mask[:,:,:228]
mask_flip = np.flip(mask[:,:,:227], axis = 2)
symm_mask[:,:,228:455] = mask_flip
symm_mask = symm_mask[18:315,19:634,:455]
gi.io.save_nifti(symm_mask.astype('uint8'), folder_out + 'tissue_mask_A5_straight_rot_symm2.nii.gz')

## Clean the symmetric average MRI brain usin the symmetric mask
symm_avg = gi.io.load_nifti(folder_out + 'A5_straight_rot_symm2.nii.gz')
mask = gi.io.load_nifti(folder_out + 'tissue_mask_A5_straight_rot_symm2_man.nii.gz')
symm_mask = np.zeros(np.shape(mask))
symm_mask[:,:,:228] = mask[:,:,:228]
mask_flip = np.flip(mask[:,:,:227], axis = 2)
symm_mask[:,:,228:455] = mask_flip
symm_avg[symm_mask==0] =1

## Average bregma and lambda
bl = gi.io.load_nifti(folder_out + 'AVG_bregma1_lambda2_bspline_A5_straight_rot.nii.gz')
symm_bl = bl[18:315,19:634,:455]

## Save volumes
gi.io.save_nifti(symm_avg.astype('uint8'), folder_out + 'A5_straight_rot_symm_masked2.nii.gz')
gi.io.save_nifti(symm_mask.astype('uint8'), folder_out + 'tissue_mask_A5_straight_rot_symm_final2.nii.gz')
gi.io.save_nifti(symm_bl.astype('uint8'), folder_out + 'AVG_bregma1_lambda2_bspline_symm2.nii.gz')


# CALCULATE DISTANCES FROM AVG BREGMA/LAMBDA FOR ALL DATAPOINTS ################

avg_points = gi.io.load_nifti(folder_out + 'AVG_bregma1_lambda2_bspline_A5.nii.gz')
avg_bregma = np.argwhere(avg_points==1)
avg_lambda = np.argwhere(avg_points==2)

z_dist_breg=[]
y_dist_breg=[]
x_dist_breg=[]

z_dist_lam=[]
y_dist_lam=[]
x_dist_lam=[]

breg_lam_dist = []

## Calculate average distance of bregma/lambda points from the average bregma/lambda
for n in range(len(sample_id)):
    points = gi.io.load_nifti(folder_in + sample_id[n] +'_bregma1_lambda2_bspline_A5.nii.gz')
    if 1 in points:
        point_bregma = np.argwhere(points==1)
        dist_bregma = (avg_bregma - point_bregma)**2
        z_dist_breg.append(dist_bregma[0][0])
        y_dist_breg.append(dist_bregma[0][1])
        x_dist_breg.append(dist_bregma[0][2])

    if 2 in points:
        point_lambda = np.argwhere(points==2)
        dist_lambda = (avg_lambda - point_lambda)**2
        z_dist_lam.append(dist_lambda[0][0])
        y_dist_lam.append(dist_lambda[0][1])
        x_dist_lam.append(dist_lambda[0][2])

    # For calculating average bregma-lambda distance in y and its stadanrd deviation
    if 1 in points and 2 in points:
        point_bregma = np.argwhere(points==1)
        point_lambda = np.argwhere(points==2)
        temp_dist_breg_lam = ((point_lambda[0][1]-point_bregma[0][1])**2+(point_lambda[0][0]-point_bregma[0][0])**2+(point_lambda[0][2]-point_bregma[0][2])**2)**(0.5)
        breg_lam_dist.append(temp_dist_breg_lam)

stdev_bregma = [np.sqrt(np.sum(z_dist_breg)/len(z_dist_breg)), np.sqrt(np.sum(y_dist_breg)/len(y_dist_breg)), np.sqrt(np.sum(x_dist_breg)/len(x_dist_breg))]
stdev_lambda = [np.sqrt(np.sum(z_dist_lam)/len(z_dist_lam)), np.sqrt(np.sum(y_dist_lam)/len(y_dist_lam)), np.sqrt(np.sum(x_dist_lam)/len(x_dist_lam))]

# Make dataframe with essential numbers
d = {'metric': ['avg bregma coord', 'avg lambda coord', 'stdev to bregma', 'stdev to lambda'], 'z': [avg_bregma[0][0], avg_lambda[0][0], stdev_bregma[0], stdev_lambda[0]],
'y' : [avg_bregma[0][1], avg_lambda[0][1], stdev_bregma[1], stdev_lambda[1]], 'x': [avg_bregma[0][2], avg_lambda[0][2], stdev_bregma[2], stdev_lambda[2]]}
df = pd.DataFrame(data=d)
df.to_csv(folder_out + 'lambda_bregma_mean_stdev.csv')


## Calculate bregma-lambda distances of every individual skull from average bregma-average lambda distance
avg_breg_lam_dist = np.sum(breg_lam_dist)/len(breg_lam_dist)
print(avg_breg_lam_dist)

dif_avg_breg_lam_dist = []
for j in range(0,len(breg_lam_dist)):
    temp_dif_avg_breg_lam_dist = (avg_breg_lam_dist - breg_lam_dist[j])**2
    dif_avg_breg_lam_dist.append(temp_dif_avg_breg_lam_dist)

stdev_breg_lam_dist =  np.sqrt(np.sum(dif_avg_breg_lam_dist)/len(dif_avg_breg_lam_dist))

# Make dataframe with essential numbers
d = {'metric': ['avg dist bregma lambda', 'stdev dist bregma lambda'], 'value': [avg_breg_lam_dist,stdev_breg_lam_dist ]}
df = pd.DataFrame(data=d)
df.to_csv(folder_out + 'lambda_bregma_distance_mean_stdev.csv')


### MAKE COORDINATE SYSTEM ####################################################

avg = gi.io.load_nifti(folder_out + 'A5_straight_rot_symm_masked2.nii.gz')

avg_points = gi.io.load_nifti(folder_out + 'AVG_bregma1_lambda2_bspline_symm2.nii.gz')
avg_bregma = np.argwhere(avg_points==1)[0]

# coordinates in x-dimension
x_arr = np.linspace(0.0125,0.025*int(np.floor(avg.shape[2]/2)),num=int(np.floor(avg.shape[2]/2)))
x_arr_backwards = x_arr[::-1]
x_concat = np.concatenate((x_arr_backwards,[0],x_arr),axis=0)
x_coords = np.tile(x_concat,(avg.shape[0],avg.shape[1],1))
gi.io.save_nifti(x_coords, folder_out + 'x_coords_MRI.nii.gz')

# coordinates in z-dimension
z_arr = np.linspace(0.0125,0.025*avg_bregma[0],num=avg_bregma[0])
z_arr_neg =  np.linspace(-0.025*(avg.shape[0]-avg_bregma[0]-1),-0.0125, num=(avg.shape[0]-avg_bregma[0]-1))
z_concat = np.concatenate((z_arr_neg,[0],z_arr),axis=0)
z_concat = z_concat[::-1]
z_coords = np.tile(z_concat,(avg.shape[2],avg.shape[1],1))
z_coords = np.swapaxes(z_coords,0,2)
gi.io.save_nifti(z_coords, folder_out + 'z_coords_MRI.nii.gz')

# coordinates in y-dimension
y_arr_neg = np.linspace(-0.025*(avg.shape[1]-avg_bregma[1]-1),-0.0125,num=(avg.shape[1]-avg_bregma[1]-1))
y_arr =  np.linspace(0.0125,0.025*avg_bregma[1], num=avg_bregma[1])
y_concat = np.concatenate((y_arr_neg,[0],y_arr),axis=0)
y_concat = y_concat[::-1]
y_coords = np.tile(y_concat,(avg.shape[0],avg.shape[2],1))
y_coords = np.swapaxes(y_coords,1,2)
gi.io.save_nifti(y_coords, folder_out + 'y_coords_MRI.nii.gz')


# REMOVE COORDINATES OUTSIDE TISSUE ################################################

# Coordinates only where tissue is
x_coords = gi.io.load_nifti(folder_out + 'x_coords_MRI.nii.gz')
y_coords = gi.io.load_nifti(folder_out + 'y_coords_MRI.nii.gz')
z_coords = gi.io.load_nifti(folder_out + 'z_coords_MRI.nii.gz')
mask = gi.io.load_nifti(folder_out + 'tissue_mask_A5_straight_rot_symm_final2.nii.gz')

x_coords[mask==0]=0
y_coords[mask==0]=0
z_coords[mask==0]=0

coords_all = np.stack([z_coords,y_coords,x_coords], axis=3)
gi.io.save_nifti(coords_all, folder_out + 'coords_all_MRI_tissue.nii.gz')

gi.io.save_nifti(x_coords, folder_out + 'x_coords_MRI_tissue.nii.gz')
gi.io.save_nifti(y_coords, folder_out + 'y_coords_MRI_tissue.nii.gz')
gi.io.save_nifti(z_coords, folder_out + 'z_coords_MRI_tissue.nii.gz')