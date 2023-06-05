### Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import scipy.ndimage
import voltools

#### Import custom libraries
import sys
sys.path.insert(0, '/opt/imaging/libraries')
import GubraImg as gi

# Path input/output
data_path = r'/CT_2_MRI/mri_mapped_ct'

# Read in the sample
id =r'ID001'

# Load the ct skull volume
V = gi.io.load_nifti(os.path.join(data_path,id+r'_ct_ccfv3_rig.nii.gz'))

# # Visualize the skull
# voltools.show_vol(V, cmap = 'jet')

### FOR X-Y direction
# Cumulative sum of thresholded volume - threshold value and layers found by visual inspection
t_val = 1 # threshold for finding the skull (between skull and air)
f_z = 170 # from z depth
t_z = 317 # to z depth

# cumulative sum
cV = np.cumsum(V[f_z:t_z,:,:].astype(np.float) > t_val, axis = 0)

# # Show the cumulative sum
# voltools.show_vol(cV.transpose((0,1,2)), cmap = 'jet')
# plt.imshow(cV[106,:,:], cmap = 'jet')

# # Visualize threshold
# voltools.show_vol(cV.transpose((0,1,2))<1, cmap = 'plasma')

# Compute depth image
sigma = 3
im_d = np.sum(cV < 1, axis = 0).astype(np.float)
im_d_g = scipy.ndimage.gaussian_filter(im_d,sigma)

# # Visualize depth images
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(im_d) # depth image (gives the surface of the skull, and can be used as the z-coordinate)
# ax[1].imshow(im_d_g) # smooth depth image

# Extract surface by sampling along the z-axis
r = 1 # number of lines to sample above and below
nx, ny = V.shape[1:3]
h = im_d_g.astype(np.int)
U = np.fromfunction(lambda z,x,y: V[h[x,y]+f_z+z-r,x,y], (r*2+1,nx,ny), dtype=int)

# # Show the sample volume
# voltools.show_vol(U.transpose((0,1,2)), cmap = 'plasma', vmax = 100)
# plt.imshow(U[4,:,:], cmap = 'plasma', vmax = 100)

# Compute the max projection and smooth
thres = 2.5 # threshold for finding the sutures
sigma = 0.5
im_U = np.max(U,axis=0).astype(np.float)
im_U_g = scipy.ndimage.gaussian_filter(im_U, sigma)

# Visualize maximum projection images
fig, ax = plt.subplots(2,2, sharex = True, sharey = True)
ax[0][0].imshow(im_U) # max projection
ax[0][1].imshow(im_U_g) # smooth max projection
ax[1][0].imshow(im_U<thres) # max projection
ax[1][1].imshow(im_U_g<thres) # smooth max projection

# Save suture image and create mask manually
gi.io.save_nifti((im_U_g<thres).astype(np.uint8)*255,os.path.join(data_path, id+r'_ct_ccfv3_rig_sutures.nii.gz'))


### CHECK RESULTS ##############################################################################

#%% Load in the mask and display the sutures
mask_sutures = np.squeeze(gi.io.load_nifti(os.path.join(data_path, id+r'_ct_rig_sutures_mask.nii.gz')))
mask = np.zeros([mask_sutures.shape[0],mask_sutures.shape[1],3])
mask[:,:,0] = mask_sutures
mask[:,:,1] = mask_sutures
mask[:,:,2] = mask_sutures
im_thres = im_U<thres
im_suture_r = im_thres*(mask[:,:,0] == 1)
im_suture_g = im_thres*(mask[:,:,1] == 2)
im_suture_b = im_thres*(mask[:,:,2] == 3)

nr,nc = im_U.shape
im_rgb = np.zeros((nr,nc,3)).astype(np.uint8)
im_rgb[:,:,0] = np.minimum(2*im_U/256 + 150*im_suture_r.astype(np.uint8),255)
im_rgb[:,:,1] = np.minimum(2*im_U/256 + 150*im_suture_g.astype(np.uint8),255)
im_rgb[:,:,2] = np.minimum(2*im_U/256 + 150*im_suture_b.astype(np.uint8),255)


fig,ax = plt.subplots(2,2)
ax[0][0].imshow(im_rgb)
ax[0][1].imshow(im_suture_r)
ax[1][0].imshow(im_suture_g)
ax[1][1].imshow(im_suture_b)
