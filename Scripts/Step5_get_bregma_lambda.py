### Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import scipy.ndimage

from skimage.morphology import disk, ball



import sys

### Import custom libraries
sys.path.insert(0, r'/opt/imaging/libraries')
import GubraImg as gi

# Path input/output
data_path = r'/CT_2_MRI/mri_mapped_ct'
mri_path = r'/CT_2_MRI/mri_brains_aligned'

# %% Import a suture mask
id =r'ID001'

# Horizontal suture mask for both hemispheres
mask_top = np.squeeze(gi.io.load_nifti(os.path.join(data_path,id+r'_ct_ccfv3_rig_sutures_mask.nii.gz')))
edges_top = np.squeeze(gi.io.load_nifti(os.path.join(data_path,id+r'_ct_ccfv3_rig_sutures.nii.gz')))
ct_vol = gi.io.load_nifti(os.path.join(data_path,id+r'_ct_ccfv3_rig.nii.gz'))
mri_vol = gi.io.load_nifti(os.path.join(mri_path,id+r'_ccfv3_rig.nii.gz'))

# %% Extract sutures under mask
sut_left_right_top = np.zeros(edges_top.shape, dtype='uint8')
sut_left_right_top[edges_top*mask_top==255]=1

sut_middle_top = np.zeros(edges_top.shape, dtype='uint8')
sut_middle_top[edges_top*mask_top==255*2]=1

sut_lamboid = np.zeros(edges_top.shape, dtype='uint8')
sut_lamboid[edges_top*mask_top==255*3]=1


# %% CORONAL SUTURE

# Get coordinates of suture elements (nonzero elements) - left and right
coords_left_right_top = np.nonzero(sut_left_right_top)
coords_left_right_top_x = coords_left_right_top[1]
coords_left_right_top_y = coords_left_right_top[0]

# Fit the coordinates: correct version
fy_bregma = np.polyfit(coords_left_right_top_x, coords_left_right_top_y,2)
py_bregma = np.poly1d(fy_bregma)

# ## Plotting  
# fig = plt.figure(figsize=(12,5))
# ax1=fig.add_subplot(131)
# ax2=fig.add_subplot(132)
# ax3=fig.add_subplot(133)
# ax1.imshow(edges_top, cmap = 'gray')
# ax2.imshow(sut_left_right_top, cmap = 'gray')
# xcoords = np.arange(75, 370, 1) #np.linspace(80, 370, 1) 
# ax3.plot(coords_left_right_top_x, coords_left_right_top_y, '.',xcoords,py_bregma(xcoords), 'r--',markersize = 1)
# ax3.set_xlim([0, 456])
# ax3.set_ylim([668, 0])
# for ax in [ax1,ax2,ax3]:
#     ax.set_aspect('equal')
#     ax.set_ylabel('y')
#     ax.set_xlabel('x')
    
    
# %% SAGITTAL SUTURE 

# Get coordinates of suture elements (nonzero elements) - left and right
coords_middle_top = np.nonzero(sut_middle_top)
coords_middle_top_x = coords_middle_top[1]
coords_middle_top_y = coords_middle_top[0]

# Fit the coordinates: correct version
fx_bregma = np.polyfit(coords_middle_top_y, coords_middle_top_x,1)
px_bregma = np.poly1d(fx_bregma)

# ## Plotting
# fig = plt.figure(figsize=(12,5))
# ax1=fig.add_subplot(131)
# ax2=fig.add_subplot(132)
# ax3=fig.add_subplot(133)
# ax1.imshow(edges_top, cmap = 'gray')
# ax2.imshow(sut_middle_top, cmap = 'gray')
# ycoords = np.arange(240,460,1) #np.linspace(240, 460, 10) 
# ax3.plot(coords_middle_top_x, coords_middle_top_y, '.',px_bregma(ycoords),ycoords, 'r--',markersize = 1)
# ax3.set_xlim([0, 456])
# ax3.set_ylim([668, 0])
# for ax in [ax1,ax2,ax3]:
#     ax.set_aspect('equal')
#     ax.set_ylabel('y')
#     ax.set_xlabel('x')
    

# %% LAMBOIDAL SUTURE 

# Get coordinates of suture elements (nonzero elements) - left and right
coords_lamboid = np.nonzero(sut_lamboid)
coords_lamboid_x = coords_lamboid[1]
coords_lamboid_y = coords_lamboid[0]

# Fit the coordinates: correct version
fy_lambda = np.polyfit(coords_lamboid_x, coords_lamboid_y,1)
py_lambda = np.poly1d(fy_lambda)

# ## Plotting
# fig = plt.figure(figsize=(12,5))
# ax1=fig.add_subplot(131)
# ax2=fig.add_subplot(132)
# ax3=fig.add_subplot(133)
# ax1.imshow(edges_top, cmap = 'gray')
# ax2.imshow(sut_lamboid, cmap = 'gray')
# xcoords = np.arange(70, 380, 1) 
# ax3.plot(coords_lamboid_x, coords_lamboid_y, '.',xcoords,py_lambda(xcoords), 'r--',markersize = 1)
# ax3.set_xlim([0, 456])
# ax3.set_ylim([668, 0])
# for ax in [ax1,ax2,ax3]:
#     ax.set_aspect('equal')
#     ax.set_ylabel('y')
#     ax.set_xlabel('x')
    

# %% INTERSECTION OF CORONAL AND SAGITTAL SUTURE FITS
    
# Find x
a = fy_bregma[0]*fx_bregma[0]
b = fy_bregma[1]*fx_bregma[0]-1
c = fx_bregma[1]+fy_bregma[2]*fx_bregma[0]
x1 = (-b + np.sqrt(b**2 - 4*a*c))/2/a
x2 = (-b - np.sqrt(b**2 - 4*a*c))/2/a

for xval in [x1, x2]:
    if xval>150 and xval<350:
        x_bregma=xval
        
# Find y
y_bregma = (x_bregma-fx_bregma[1])/fx_bregma[0]

x_bregma = np.rint(np.around(x_bregma)).astype('uint16')
y_bregma = np.rint(np.around(y_bregma)).astype('uint16')

# # Plot both fits together
# fig = plt.figure(figsize=(14,8))
# ax1=fig.add_subplot(131)
# ax2=fig.add_subplot(132)
# ax3=fig.add_subplot(133)
# ax1.imshow(edges_top, cmap = 'gray')
# ax2.imshow(sut_middle_top + sut_left_right_top, cmap = 'gray')
# ax3.imshow(sut_middle_top + sut_left_right_top, cmap = 'gray')
# ycoords = np.arange(240,460,1) #np.linspace(240, 460, 10) 
# ax3.plot(px_bregma(ycoords),ycoords, 'r--',markersize = 1)
# xcoords = np.arange(75, 370, 1) #np.linspace(80, 370, 1) 
# ax3.plot(xcoords,py_bregma(xcoords), 'r--',markersize = 1)
# ax3.plot(x_bregma,y_bregma, 'bo',markersize = 3, label = 'x = '+ str(x_bregma)+', y = '+ str(y_bregma))
# ax3.set_xlim([0, 456])
# ax3.set_ylim([668, 0])
# ax3.legend()
# for ax in [ax1,ax2,ax3]:
#     ax.set_aspect('equal')
#     ax.set_ylabel('y')
#     ax.set_xlabel('x')


# %% Find Z FOR BREGMA

# Plot intensity profile in z-direction for x/y coordinate
m = 10
n = 10
ct_z = ct_vol[:,y_bregma-n:y_bregma+n,x_bregma-n:x_bregma+n]
ct_z_mean = np.mean(ct_z,axis = (1,2))

skull_thresh = 8
idx_thresh = np.argwhere(ct_z_mean > skull_thresh).tolist()

for z_idx in reversed(idx_thresh):
    if z_idx[0]>200:
        z_bregma=z_idx[0]
        break

# # Visualize
# fig = plt.figure(figsize=(6,5))
# plt.plot(np.linspace(0,320,320),ct_z_mean)
# plt.axvline(z_bregma,color='red', ymin = 0,ymax=np.max(ct_z_mean), label = 'z = '+ str(z_bregma))
# plt.legend()
# plt.xlabel('z')
# plt.ylabel('Signal intensity')
# fig.savefig(os.path.join(data_path, id+r'_bregma_z.png'))


# %% INTERSECTION OF LAMBOIDAL AND SAGITTAL SUTURE FITS
    
# Find x
x_lambda = (fx_bregma[1]+fy_lambda[1]*fx_bregma[0])/(1-fx_bregma[0]*fy_lambda[0])
        
# Find y
y_lambda = x_lambda*fy_lambda[0]+fy_lambda[1]

x_lambda = np.rint(np.around(x_lambda)).astype('uint16')
y_lambda = np.rint(np.around(y_lambda)).astype('uint16')

# # Plot both fits together
# fig = plt.figure(figsize=(14,8))
# ax1=fig.add_subplot(131)
# ax2=fig.add_subplot(132)
# ax3=fig.add_subplot(133)
# ax1.imshow(edges_top, cmap = 'gray')
# ax2.imshow(sut_middle_top + sut_lamboid, cmap = 'gray')
# ax3.imshow(sut_middle_top + sut_lamboid, cmap = 'gray')
# ycoords = np.arange(240,500,1) 
# ax3.plot(px_bregma(ycoords),ycoords, 'r--',markersize = 1)
# xcoords = np.arange(70, 380, 1) 
# ax3.plot(xcoords,py_lambda(xcoords), 'r--',markersize = 1)
# ax3.plot(x_lambda,y_lambda, 'bo',markersize = 3, label = 'x = '+ str(x_lambda)+', y = '+ str(y_lambda))
# ax3.set_xlim([0, 456])
# ax3.set_ylim([668, 0])
# ax3.legend()
# for ax in [ax1,ax2,ax3]:
#     ax.set_aspect('equal')
#     ax.set_ylabel('y')
#     ax.set_xlabel('x')


# %% Find Z FOR LAMBDA

# Plot intensity profile in z-direction for x/y coordinate
m = 5
n = 10
ct_z = ct_vol[:,y_lambda-n:y_lambda+n,x_lambda-n:x_lambda+n]
ct_z_mean = np.mean(ct_z,axis = (1,2))

skull_thresh = 8
idx_thresh = np.argwhere(ct_z_mean > skull_thresh).tolist()

for z_idx in reversed(idx_thresh):
    if z_idx[0]>200:
        z_lambda=z_idx[0]
        break

# # Visualize
# fig = plt.figure(figsize=(6,5))
# plt.plot(np.linspace(0,320,320),ct_z_mean)
# plt.axvline(z_lambda,color='red', ymin = 0,ymax=np.max(ct_z_mean), label = 'z = '+ str(z_lambda))
# plt.legend()
# plt.xlabel('z')
# plt.ylabel('Signal intensity')
# fig.savefig(os.path.join(data_path, id+r'_lambda_z.png'))


# %% Save as a segmentation volume
segm = np.zeros(np.shape(mri_vol))
segm[z_bregma,y_bregma,x_bregma]=1
segm[z_lambda,y_lambda,x_lambda]=2
gi.io.save_nifti(segm, os.path.join(data_path, id+r'_bregma1_lambda2.nii.gz'))