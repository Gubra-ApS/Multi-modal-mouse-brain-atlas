import os, sys, stat
from shutil import move
import glob
import time
import shutil
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, color, img_as_uint, filters, io
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.ndimage import rotate, median_filter, binary_dilation,generate_binary_structure
import pandas as pd
import skimage.external.tifffile
import skimage.transform

#### Import custom libraries
import sys
sys.path.insert(0, '/opt/imaging/libraries')

import importlib
import GubraImg as gi
importlib.reload(gi)
####


### Set parameters ###################################################################
folder_input = '/MRI_data/Data' # Raw data INPUT
folder_input_ct = '/CT_mouse_skull_study' # Raw data INPUT
folder_output_mri = '/CT_2_MRI/mri' # USER OUTPUT: mri
folder_output_masked = '/CT_2_MRI/masked_brain_mri' # USER OUTPUT: masked brains
folder_output_masks = '/CT_2_MRI/brain_masks' # USER OUTPUT: tissue masks
folder_output_reg = '/CT_2_MRI/mri_mapped_ct' # USER OUTPUT: mapped ct
mask_file = 'Mask.nii'
anat_file = '_T2_TRUE_FISP_3D_.nii'

if not os.path.exists( folder_output_masked): os.makedirs( folder_output_masked )
if not os.path.exists( folder_output_reg): os.makedirs(  folder_output_reg)
if not os.path.exists( folder_output_mri): os.makedirs( folder_output_mri )


# additional parameters
ID_tag_length = 3
group_tag_length = 3

# USER DEFINED PARAMETERS
voxel_size_xy_lsfm =  4.7944
voxel_size_z_lsfm =  10
voxel_size_xyz_ct =  22.578
voxel_size_xyz_mri =  78

#####################################################################################
# Functions

def load_tiffs(scan_folder, channels=['C00'], scale_xy=1, scale_z=1, range_z=[]):
    '''
    Load .tiff stack from scan folder into an ndarray

    Call:
        load_tiffs(scan_folder, channels=['C00'], scale_xy=1, scale_z=1, range_z=[]):

    Args:
        scan_folder (string)       : e.g. '/home/user/data/volume.nii.gz'
        channels (list of strings) : channel(s) to load, e.g. ['C00', 'C01']
        scale_xy (float)           : scale factor in xy-direction
        scale_z (float)            : scale factor in z-direction
        range_z (2x1 ndarray)      : slices to read [first_slice, last_slice]

    Returns:
        vol_out (ndarray) : 3D (4D in case of multiple channels) volume of entire .tiff stack
    '''

    # read files for first tag
    file_search = sorted( glob.glob( os.path.join(scan_folder, '*' + channels[0] + '*.tif') ) )

    # only read select files if range is provided
    if len(range_z) > 0:
        file_search = file_search[range_z[0]:range_z[1]]
    n_files = len(file_search)

    # read dummy image and determine output dimensionality
    dummy = skimage.external.tifffile.imread(file_search[0], multifile=False)
    n_rows, n_cols = skimage.transform.rescale(dummy, scale=scale_xy).shape
    n_slices = skimage.transform.rescale( np.zeros(n_files), scale_z ).shape[0]
    n_tags = len(channels)
    print(n_slices)

    # read files for current channel
    file_search = sorted( glob.glob( os.path.join(scan_folder, '*'  + '*.tif') ) )
    if len(range_z) > 0:
        file_search = file_search[range_z[0]:range_z[1]]

    # initialise current volume
    vol_tmp = np.zeros( (n_files, n_rows, n_cols), dtype = 'uint16')

    # load files into current volume
    for j, file in enumerate(file_search):
        print(file)
        im = skimage.external.tifffile.imread(file, multifile=False)
        #im = np.log(skimage.io.imread(file))*1000
        im = im.astype('uint16')
        if scale_xy != 1: # rescale xy if needed
            im = skimage.transform.rescale(im, scale=scale_xy, order = 3, preserve_range=True, multichannel=False)
            im = im.astype('uint16')
        vol_tmp[j] = im

    # rescale z-direction
    if scale_z != 1:
        vol_tmp = skimage.transform.rescale(vol_tmp, scale=(scale_z,1,1), order = 3, preserve_range=True, multichannel=False)

    # finally return output volume
    return vol_tmp


def bounds_per_dimension(ndarray):
    return map(
        lambda e: range(e.min(), e.max() + 1),
        np.where(ndarray > 2)
    )

def zero_trim_ndarray(ndarray):
    return ndarray[np.ix_(*bounds_per_dimension(ndarray))]

####################################################################################

# Turn off plots
plt.ioff()


# Search for scan folders
mr_tags = ['M0572','M0573','M0574','M0576','M0579','M0588','M0593','M0594','M0595','M0597','M0598','M0599']
mr_tags2 = ['M0572','MO573','MO574','MO576','MO579','MO588','MO593','MO594','MO595','MO597','MO598','MO599']
id_tags = ['ID012','ID007','ID011','ID008','ID010','ID002','ID004','ID003','ID001','ID005','ID006','ID009']

masks = []
for i in range(0,len(id_tags)):
    masks.append(os.path.join(folder_input,mr_tags[i],'DICOM2NIFTI',mask_file))

t2_fisp = []
for i in range(0,len(id_tags)):
    t2_fisp.append(os.path.join(folder_input,mr_tags[i],'DICOM2NIFTI',mr_tags2[i]+'-'+id_tags[i]+anat_file))

cts = []
for i in range(0,len(id_tags)):
    ct_path = gi.io.search(folder_input_ct+'/'+id_tags[i],'ID*')
    cts.append(os.path.join(ct_path[0],'LFOV-50kV-LE2-2s-22.6micro_TIFFstack'))

# Make nifti files from CT tiff-STACK, upsample to 25 micron resolution
for nr, fldr in enumerate(cts):
    if os.path.isdir(fldr):
        tag   = 'ID' + fldr[ (fldr.find('ID') + 2):(fldr.find('ID') + ID_tag_length + 2) ]
        print('Now analysing: ' + tag)

        # Read raw data
        print('Reading raw tiffs..')
        vol = load_tiffs(fldr, channels=['Z'], scale_xy=(voxel_size_xyz_ct/25), scale_z=(voxel_size_xyz_ct/20))
        print(vol.shape)
        gi.io.save_nifti(vol.astype('uint16'), folder_input_ct +'/'+ tag + '_ct.nii.gz')


# Upsample to 25 um resolution and Mask the brains from MRI images
for nr, fldr in enumerate(t2_fisp):
    print(fldr)
    temp_t2 = gi.io.load_nifti(t2_fisp[nr])
    temp_t2 = np.around(256*(temp_t2-np.min(temp_t2))/(np.max(temp_t2)-np.min(temp_t2))).astype('uint8')
    temp_mask= gi.io.load_nifti(masks[nr]).astype('uint8')

    # rescale mri
    masked_t2 = np.copy(temp_t2)
    masked_t2[temp_mask==1]=0
    temp_t2_masked = skimage.transform.rescale(masked_t2, scale=voxel_size_xyz_mri/25, preserve_range=True, multichannel=False).astype('uint8')
    temp_t2_scaled = skimage.transform.rescale(temp_t2, scale=voxel_size_xyz_mri/25, preserve_range=True, multichannel=False).astype('uint8')

    # rescale mask
    masked_t2 = np.copy(temp_t2)
    masked_t2[temp_mask==0]=0
    masked_t2_scaled = skimage.transform.rescale(masked_t2, scale=voxel_size_xyz_mri/25, preserve_range=True, multichannel=False).astype('uint8')
    masked_t2_scaled[masked_t2_scaled >0]=1

    gi.io.save_nifti(temp_t2_masked.astype('uint8'), folder_output_masked +'/'+ id_tags[nr] + '_T2_masked_brain.nii.gz')
    gi.io.save_nifti(temp_t2_scaled.astype('uint8'), folder_output_mri +'/'+ id_tags[nr] + '_T2.nii.gz')
    gi.io.save_nifti(masked_t2_scaled.astype('uint8'), folder_output_masks +'/'+ id_tags[nr] + '_brain_mask.nii.gz')


#Orient mri volumes to standard direction
for i in range(0,len(id_tags)):
    print(id_tags[i])
    temp_t2 = gi.io.load_nifti(folder_output_mri +'/'+ id_tags[i] + '_T2.nii.gz')
    print(temp_t2.shape)
    temp_t2 = np.swapaxes(temp_t2,0,1) # for IDs 12, 11, 8, 10
    temp_t2_rot = rotate(temp_t2,180, axes = (0, 2)) # for IDs 12, 11, 8, 10
    #temp_t2_rot = np.swapaxes(temp_t2,0,1) #  for ID5, ID6 and ID9, ID2, ID4, ID3, ID1
    print(temp_t2_rot.shape)
    gi.io.save_nifti(temp_t2_rot.astype('uint8'), folder_output_mri +'/'+ id_tags[i] + '_T2_rot.nii.gz')

# cut mr volumes
for i in range(0,len(id_tags)):
    temp_t2 = gi.io.load_nifti(folder_output_mri +'/'+ id_tags[i] + '_T2_rot.nii.gz')
    temp_t2_cut = temp_t2[:,0:895,:]
    temp_t2_cut2 = temp_t2_cut[:,30:880,165:805]
    # # ID002
    # temp_t2_cut2 = temp_t2[:,30:935,175:815]
    # #ID003
    # temp_t2_cut2 = temp_t2[:,30:935,165:805]
    gi.io.save_nifti(temp_t2_cut2.astype('uint8'), folder_output_mri +'/'+ id_tags[i] + '_T2_rot_cut.nii.gz')


# Orient mri masks to standard direction
for i in range(0,12):
    print(id_tags[i])
    temp_t2 = gi.io.load_nifti(folder_output_masks +'/'+ id_tags[i] + '_brain_mask.nii.gz')
    if i < 5:
        temp_t2 = np.swapaxes(temp_t2,0,1) # for IDs 12, 11, 8, 10
        temp_t2_rot = rotate(temp_t2,180, axes = (0, 2)) # for IDs 12, 11, 8, 10
    else:
        temp_t2_rot = np.swapaxes(temp_t2,0,1) #  for ID5, ID6 and ID9, ID2, ID4, ID3, ID1
    temp_t2_rot = temp_t2_rot[:,0:895,:]
    temp_t2_cut = temp_t2_rot[:,30:880,165:805]
    # # ID002
    # temp_t2_cut = temp_t2_rot[:,30:935,175:815]
    # #ID003
    # temp_t2_cut = temp_t2_rot[:,30:935,165:805]
    gi.io.save_nifti(temp_t2_cut.astype('uint8'), folder_output_masks +'/'+ id_tags[i] + '_brain_mask_rot_cut.nii.gz')


#Create skull masks from MRI images for aligning CT with MRI
for i in range(0,len(id_tags)):
    temp_t2 = gi.io.load_nifti(folder_output_mri +'/'+ id_tags[i] + '_T2_rot_cut.nii.gz')
    brain_mask = gi.io.load_nifti(folder_output_masks +'/'+ id_tags[i] + '_brain_mask_rot_cut.nii.gz')
    struct = generate_binary_structure(3, 3)
    brain_mask_dil = binary_dilation(brain_mask, structure=struct, iterations=23).astype(brain_mask.dtype)
    thresh = filters.threshold_otsu(temp_t2)
    mask_bone = temp_t2 <= thresh
    mask_bone[brain_mask_dil==0]=0
    gi.io.save_nifti(mask_bone.astype('uint8'), folder_output_mri +'/'+ id_tags[i] + '_skull_mask.nii.gz')



# # Orient CT volumes to standard direction
i=0
temp_t2 = gi.io.load_nifti(folder_input_ct +'/'+ id_tags[i] + '_ct.nii.gz')
print(id_tags[i])
print(temp_t2.shape)
# # #ID7
# temp_t2 = rotate(temp_t2,92, axes = (1, 2))
# temp_t2_rot = np.swapaxes(temp_t2,0,1)
# temp_t2_rot = rotate(temp_t2_rot,-170, axes = (0, 2))
# #ID12
# temp_t2 = np.swapaxes(temp_t2,0,1)
# temp_t2_rot = rotate(temp_t2,190, axes = (0, 2))
# #ID11
# temp_t2 = rotate(temp_t2,85, axes = (1, 2))
# temp_t2 = np.swapaxes(temp_t2,0,1)
# temp_t2_rot = rotate(temp_t2,180, axes = (0, 2))
# #ID008
# temp_t2 = rotate(temp_t2,-95, axes = (1, 2))
# temp_t2 = np.swapaxes(temp_t2,0,1)
# temp_t2_rot = rotate(temp_t2,170, axes = (0, 2))
# # ID010
# temp_t2 = rotate(temp_t2,135, axes = (1, 2))
# temp_t2 = np.swapaxes(temp_t2,0,1)
# temp_t2_rot = rotate(temp_t2,180, axes = (0, 2))
# # # ID002
# temp_t2 = rotate(temp_t2,-72, axes = (1, 2))
# temp_t2_rot = np.swapaxes(temp_t2,0,1)
# temp_t2_rot = rotate(temp_t2_rot,-182, axes = (0, 2))
# # ID004
# temp_t2 = rotate(temp_t2,90, axes = (1, 2))
# temp_t2_rot = np.swapaxes(temp_t2,0,1)
# temp_t2_rot = rotate(temp_t2_rot,-180, axes = (0, 2))
# # ID003
# temp_t2 = rotate(temp_t2,110, axes = (1, 2))
# temp_t2_rot = np.swapaxes(temp_t2,0,1)
# temp_t2_rot = rotate(temp_t2_rot,-180, axes = (0, 2))
# ID001
temp_t2 = rotate(temp_t2,-90, axes = (1, 2))
temp_t2_rot = np.swapaxes(temp_t2,0,1)
temp_t2_rot = rotate(temp_t2_rot,-175, axes = (0, 2))
# # ID005
# temp_t2 = rotate(temp_t2,135, axes = (1, 2))
# temp_t2_rot = np.swapaxes(temp_t2,0,1)
# temp_t2_rot = rotate(temp_t2_rot,-90, axes = (0, 2))
# # ID006
# temp_t2 = rotate(temp_t2,-15, axes = (1, 2))
# temp_t2_rot = np.swapaxes(temp_t2,0,1)
# temp_t2_rot = rotate(temp_t2_rot,-175, axes = (0, 2))
# # ID009
# temp_t2 = rotate(temp_t2,135, axes = (1, 2))
# temp_t2_rot = np.swapaxes(temp_t2,0,1)
# temp_t2_rot = rotate(temp_t2_rot,-160, axes = (0, 2))
#
temp_t2_rot_cut = zero_trim_ndarray(temp_t2_rot)
temp_t2_rot_cut = temp_t2_rot_cut[60:,275:,:]
temp_t2_rot_cut = temp_t2_rot_cut[50:,:,:]
gi.io.save_nifti(temp_t2_rot_cut.astype('uint16'),folder_input_ct +'/'+ id_tags[i] + '_ct_rot_cut.nii.gz')


# Register CT skulls to MRI skulls
for i in range(0,len(id_tags)):
    tag = id_tags[i]
    print('Now analysing: ' + tag)
    ####  REGISTRATION (ct -> mri)
    moving = folder_input_ct +'/'+ tag + '_ct_rot_cut.nii.gz'
    fixed = folder_output_mri +'/'+ tag + '_skull_mask.nii.gz'
    result_path = folder_output_reg
    elastix_path = '/elastix'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,result_path)
    # rigid Registration
    #def registration(params, result_name, init_trans=r'', f_mask=r'', save_nifti=True, datatype='uint16'):
    hu.registration('Rigid_mri2ct_Jan2021.txt', tag+'_ct_rig')


# # Register CT skulls to MRI skulls step 2 - afetr MRI brains have been mapped to AIBS template (see Step3_average_variational.py)
# Path to ccfv3 atlas
cffv3_dir = r'/.../ccfv3_25um'
folder_output_mri_aligned = '/CT_2_MRI/mri_brains_aligned/'


for i in range(0,len(id_tags)):
    tag = id_tags[i]
    print('Now analysing: ' + tag)
    ####  REGISTRATION (ct -> mri)
    moving = folder_output_reg +'/'+ tag + '_ct_rig.nii.gz'
    fixed = cffv3_dir + r'/ccfv3_template_olf_pad.nii.gz'
    result_path = folder_output_reg
    elastix_path = '/elastix'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,result_path)
    # Transform CT volumes
    hu.transform_vol(moving, folder_output_mri_aligned+tag+'_ccfv3_rig', tag+'_ct_ccfv3_rig', type='vol')

# Flip skulls in coronal orientation for ID2
for i in range(0,len(id_tags)):
    tag = id_tags[i]
    print('Now analysing: ' + tag)
    if tag =='ID002':
        print('Now flipping: ' + tag)
        temp = gi.io.load_nifti(folder_output_reg +'/'+ id_tags[i] + '_ct_ccfv3_rig.nii.gz')
        temp_flip = np.flip(temp,axis=2)
        gi.io.save_nifti(temp_flip.astype('uint16'),folder_output_reg +'/'+ id_tags[i] + '_ct_ccfv3_rig.nii.gz')
