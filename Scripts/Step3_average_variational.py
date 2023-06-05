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
from skimage.morphology import watershed, cube, binary_dilation, binary_erosion, remove_small_holes, remove_small_objects, binary_opening
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage import morphology, transform
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.restoration import (denoise_tv_chambolle)

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
#############################################################################
# functions
def ps_ball(radius):
    r"""
    Creates spherical ball structuring element for morphological operations

    Parameters
    ----------
    radius : float or int
        The desired radius of the structuring element

    Returns
    -------
    strel : 3D-array
        A 3D numpy array of the structuring element
    """
    rad = int(np.ceil(radius))
    other = np.ones((2 * rad + 1, 2 * rad + 1, 2 * rad + 1), dtype=bool)
    other[rad, rad, rad] = False
    ball = ndimage.distance_transform_edt(other) < radius
    return ball



#############################################################################
# Path input/output
folder_in = '/CT_2_MRI/'
folder_out = +'/CT_2_MRI/mri_brains_aligned/'

if not os.path.exists( folder_out): os.makedirs( folder_out )

# Selected reference brain from study
reference_brain_id = r'ID008'

# Path to ccfv3 atlas
cffv3_dir = r'/ccfv3_25um'

# elastix path
elastix_path = '/elastix'

# List of samples to include in atlas
sample_id = ['ID012','ID007','ID011','ID008','ID010','ID002','ID004','ID003','ID001','ID005','ID006','ID009']

id_path = []
for i in range(0,len(sample_id)):
    id_path.append(os.path.join(folder_in, 'mri', sample_id[i]+'_T2_rot_cut.nii.gz'))

# Masks
mask_path = []
for i in range(0,len(sample_id)):
    mask_path.append(os.path.join(folder_in,'brain_masks', sample_id[i]+'_brain_mask_rot_cut_man.nii.gz')) # Manually corrected masks sample_id[i]+'_brain_mask_rot_cut.nii.gz'

# CT skulls
ct_path = []
for i in range(0,len(sample_id)):
    ct_path.append(os.path.join(folder_in,'mri_mapped_ct', sample_id[i]+'_ct_rig.nii.gz'))

for nr, fldr in enumerate(id_path):
    print(fldr)
    mask_old = gi.io.load_nifti(mask_path[nr])
    mask_new = gi.io.load_nifti(os.path.join(folder_in,'brain_masks','ID003_brain_mask_rot_cut.nii.gz'))
    mask_shift = np.zeros(mask_new.shape)
    mask_shift[:,0:850,:] = mask_old
    mask_shift[:,850:,:] = mask_new[:,850:,:]
    gi.io.save_nifti(mask_shift.astype('uint8'), os.path.join(folder_in, 'brain_masks', 'ID003_brain_mask_rot_cut_man.nii.gz'))

# Additional mask correction
for nr, fldr in enumerate(id_path):
    print(fldr)
    mri_orig = gi.io.load_nifti(id_path[nr])
    mask = gi.io.load_nifti(mask_path[nr])
    mri =np.abs( 1-mri_orig)
    mri = ndimage.gaussian_filter(mri,sigma=2)
    mri[mri<=200]=0
    mri[mri>200]=1

    # # for ID12, ID1, ID5, ID8 different parameters for olfactory bulb than rest of the brain
    # mri_olf_mask = np.zeros(mri.shape)
    # mri_olf_mask[:,0:240,:]=1
    # mri_other_mask = np.zeros(mri.shape)
    # mri_other_mask[:,240:,:]=1
    # mri[:,240:,:][mri[:,240:,:]<=200]=0
    # mri[:,240:,:][mri[:,240:,:]>200]=1
    # mri[:,0:240,:][mri[:,0:240,:]<=235]=0
    # mri[:,0:240,:][mri[:,0:240,:]>235]=1

    mri[mri_orig<10]=0
    mri[mri_orig>30]=0

    mri = binary_dilation(mri, selem = np.ones((10,10,10))).astype('uint8')
    mri = binary_erosion(mri, selem = np.ones((8,8,8))).astype('uint8')
    clean_mask = np.copy(mask)
    clean_mask[mri==1]=0
    clean_mask = np.array(clean_mask, dtype = bool)
    clean_mask = remove_small_holes(clean_mask, area_threshold = 300, connectivity = 3)
    for i in range(clean_mask.shape[0]):
        clean_mask[i,:,:] = remove_small_objects(clean_mask[i,:,:] , min_size = 200, connectivity = 2)*1
    gi.io.save_nifti(clean_mask.astype('uint8'), os.path.join(folder_in, 'brain_masks', sample_id[nr]+'_brain_mask_corr.nii.gz'))


# Corrected masks
mask_path = []
for i in range(0,len(sample_id)):
    mask_path.append(os.path.join(folder_in,'brain_masks', sample_id[i]+'_brain_mask_corr.nii.gz'))

# Apply masks
for nr, fldr in enumerate(id_path):
    print(fldr)
    head = gi.io.load_nifti(fldr)
    mask = gi.io.load_nifti(mask_path[nr])
    brain = np.zeros(head.shape)
    brain[mask==1]=head[mask==1]
    gi.io.save_nifti(brain, os.path.join(folder_in, 'mri', sample_id[nr]+'_T2_rot_cut_masked.nii.gz'))

# zero-pad AIBS template
temp = gi.io.load_nifti(cffv3_dir + r'/ccfv3_template_olf.nii.gz')
temp_pad = np.pad(temp,((0,0),(70,70),(0,0)), mode = 'constant')
gi.io.save_nifti(temp_pad,cffv3_dir + r'/ccfv3_template_olf_pad.nii.gz')

# zero-pad AIBS ano
temp = gi.io.load_nifti(cffv3_dir + r'/ccfv3_ano_olf.nii.gz')
temp_pad = np.pad(temp,((0,0),(70,70),(0,0)), mode = 'constant')
gi.io.save_nifti(temp_pad,cffv3_dir + r'/ccfv3_ano_olf_pad.nii.gz')

# zero-pad AIBS ano encoded
temp = gi.io.load_nifti(cffv3_dir + r'/ccfv3_ano_olf_encoded.nii.gz')
temp_pad = np.pad(temp,((0,0),(70,70),(0,0)), mode = 'constant')
gi.io.save_nifti(temp_pad,cffv3_dir + r'/ccfv3_ano_olf_encoded_pad.nii.gz')


#### 1) Align to Allen atlas for symmetry
for n in range(len(sample_id)):
    moving = folder_in +'/mri/'+ sample_id[n] + '_T2_rot_cut_masked.nii.gz'
    fixed = cffv3_dir + r'/ccfv3_template_olf_pad.nii.gz'
    elastix_path = '/elastix'
    result_path = folder_out
    hu = gi.registration.Elastix(moving,fixed,elastix_path,result_path)
    # Affine Registration - GIVES GOOD GUESS OF THE ROTATION PARAMETERS TO BE USED (FIRST THREE PARAMETERS)
    hu.registration('affine_DTI.txt', sample_id[n] +'_ccfv3_rig')

# Flip ID2 in horizontal plane
img = gi.io.load_nifti( folder_out + 'ID002_ccfv3_rig.nii.gz')
img_flip = np.flip(img,axis = 2)
gi.io.save_nifti(img_flip, folder_out + 'ID002_ccfv3_rig.nii.gz')

# HISTOGRAM MATCHING #######################################################
reference_histogram = gi.io.load_nifti(os.path.join(folder_out, 'ID008_ccfv3_rig.nii.gz'))
for i in range(len(sample_id)):
    temp = gi.io.load_nifti(os.path.join(folder_out, sample_id[i]+'_ccfv3_rig.nii.gz'))
    temp = transform.match_histograms(temp.astype('float'), reference_histogram.astype('float'), multichannel=False)
    gi.io.save_nifti(temp.astype('uint8'),os.path.join(folder_out, sample_id[i]+'_ccfv3_rig_matched.nii.gz'))

# REGISTRATIONS ############################################################

#### 2) Run aff + average of all to A0
fixed = folder_out + reference_brain_id + '_ccfv3_rig_matched.nii.gz'

# do registrations
for n in range(len(sample_id)):
    moving = folder_out + sample_id[n] + '_ccfv3_rig_matched.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.registration('Affine_Gubra_June2019.txt',  sample_id[n]+'_aff_A0')

# compute average brain
temp = gi.io.load_nifti(fixed)

avg = np.zeros(temp.shape,'float')
divide_mask = np.zeros(temp.shape, 'float')

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = gi.io.load_nifti(os.path.join(folder_out, sample_id[i]+'_aff_A0.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask,out=np.zeros_like(avg), where=divide_mask!=0)
gi.io.save_nifti(avg.astype('uint8'), folder_out + 'A0.nii.gz')
print(counter)



#### 3) Run bspline + average of all to A1
fixed = folder_out + 'A0.nii.gz'

#
# do registrations
for n in range(len(sample_id)):
    print(sample_id[n])
    moving = folder_out + sample_id[n]+'_aff_A0.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.registration('Bspline_Gubra_June2019_goodsamples_step1.txt',  sample_id[n]+'_bspline_A1')

# compute average brain
temp = gi.io.load_nifti(fixed)

avg = np.zeros(temp.shape,'float')
divide_mask = np.zeros(temp.shape, 'float')

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = gi.io.load_nifti(os.path.join(folder_out, sample_id[i]+'_bspline_A1.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask,out=np.zeros_like(avg), where=divide_mask!=0)
gi.io.save_nifti(avg.astype('uint8'), folder_out + 'A1.nii.gz')
print(counter)



# #### 4) Run bspline + average of all to A2
fixed = folder_out + 'A1.nii.gz'

# do registrations
for n in range(len(sample_id)):
    moving = folder_out + sample_id[n]+'_bspline_A1.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.registration('Bspline_Gubra_June2019_goodsamples_step2.txt',  sample_id[n]+'_bspline_A2')

# compute average brain
temp = gi.io.load_nifti(fixed)

avg = np.zeros(temp.shape,'float')
divide_mask = np.zeros(temp.shape, 'float')

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = gi.io.load_nifti(os.path.join(folder_out, sample_id[i]+'_bspline_A2.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask,out=np.zeros_like(avg), where=divide_mask!=0)
gi.io.save_nifti(avg.astype('uint8'), folder_out + 'A2.nii.gz')
print(counter)


# #### 5) Run bspline + average of all to A3
fixed = folder_out + 'A2.nii.gz'
#
# do registrations
for n in range(len(sample_id)):
    moving = folder_out + sample_id[n]+'_bspline_A2.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.registration('Bspline_Gubra_June2019_goodsamples_step3.txt',  sample_id[n]+'_bspline_A3')

# compute average brain
temp = gi.io.load_nifti(fixed)

avg = np.zeros(temp.shape,'float')
divide_mask = np.zeros(temp.shape, 'float')

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = gi.io.load_nifti(os.path.join(folder_out, sample_id[i]+'_bspline_A3.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask,out=np.zeros_like(avg), where=divide_mask!=0)
gi.io.save_nifti(avg.astype('uint8'), folder_out + 'A3.nii.gz')
print(counter)

# #### 6) Run bspline + average of all to A4
fixed = folder_out + 'A3.nii.gz'

# do registrations
for n in range(len(sample_id)):
    moving = folder_out + sample_id[n]+'_bspline_A3.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.registration('Bspline_Gubra_June2019_goodsamples_step4.txt',  sample_id[n]+'_bspline_A4')

# compute average brain
temp = gi.io.load_nifti(fixed)

avg = np.zeros(temp.shape,'float')
divide_mask = np.zeros(temp.shape, 'float')

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = gi.io.load_nifti(os.path.join(folder_out, sample_id[i]+'_bspline_A4.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask,out=np.zeros_like(avg), where=divide_mask!=0)
gi.io.save_nifti(avg.astype('uint8'), folder_out + 'A4.nii.gz')
print(counter)


#### 7) Run bspline + average of all to A5
fixed = folder_out + 'A4.nii.gz'

# do registrations
for n in range(len(sample_id)):
    moving = folder_out + sample_id[n]+'_bspline_A4.nii.gz'
    hu = gi.registration.Elastix(moving,fixed,elastix_path,folder_out)
    hu.registration('Bspline_Gubra_June2019_goodsamples_step5.txt',  sample_id[n]+'_bspline_A5')

# compute average brain
temp = gi.io.load_nifti(fixed)

avg = np.zeros(temp.shape,'float')
divide_mask = np.zeros(temp.shape, 'float')

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = gi.io.load_nifti(os.path.join(folder_out, sample_id[i]+'_bspline_A5.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    # remove areas with MRI artifacts from the individual volume before averaging
    if sample_id[i] == 'ID011' or sample_id[i] == 'ID003' or sample_id[i] == 'ID006' or sample_id[i] == 'ID012':
        defect_mask = gi.io.load_nifti(os.path.join(folder_out, sample_id[i]+'_bspline_A5_mask_defect.nii.gz'))
        temp[defect_mask ==1]=0
        temp_mask[defect_mask ==1]=0
    if sample_id[i] == 'ID005':
        defect_mask = gi.io.load_nifti(os.path.join(folder_out, sample_id[i]+'_bspline_A5_mask_defect.nii.gz'))
        defect_mask[236:,:,:] = 1
        temp[defect_mask ==1]=0
        temp_mask[defect_mask ==1]=0

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask,out=np.zeros_like(avg), where=divide_mask!=0)
gi.io.save_nifti(avg.astype('uint8'), folder_out + 'A5.nii.gz')
print(counter)


# SYMMETRY AND MASKING #######################################################
avg = gi.io.load_nifti(folder_out + 'A5.nii.gz')
tissue_mask = np.zeros(np.shape(avg))
tissue_mask[avg>=40]=1
selem = morphology.ball(radius=5)
tissue_mask = binary_opening(tissue_mask, selem = selem)
gi.io.save_nifti(tissue_mask.astype('uint8'), folder_out + 'tissue_mask_A5.nii.gz') # correct then manually

symm_avg = np.zeros(np.shape(avg))
symm_avg[:,:,:228] = avg[:,:,:228]
avg_flip = np.flip(avg[:,:,:228], axis = 2)
symm_avg[:,:,227:455] = avg_flip
symm_avg = symm_avg[18:315,19:634,:455]

mask = gi.io.load_nifti(folder_out + 'tissue_mask_A5_man.nii.gz')
symm_mask = np.zeros(np.shape(mask))
symm_mask[:,:,:228] = mask[:,:,:228]
mask_flip = np.flip(mask[:,:,:228], axis = 2)
symm_mask[:,:,227:455] = mask_flip
symm_mask = symm_mask[18:315,19:634,:455]

symm_avg[symm_mask==0] =1

gi.io.save_nifti(symm_avg.astype('uint8'), folder_out + 'A5_symm_masked.nii.gz')
gi.io.save_nifti(symm_mask.astype('uint8'), folder_out + 'tissue_mask_A5_symm.nii.gz')