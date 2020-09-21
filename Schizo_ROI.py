import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import nilearn
from nilearn import plotting
from nilearn import image
from nilearn import datasets
import nibabel as nib
import matplotlib.image as mpimg
import h5py
import imageio
import scipy.misc as spmi
import nibabel as nib
from nilearn.image import get_data
import os
import random
from random import seed
from nideconv.utils import roi
from itertools import chain
from nilearn.regions import Parcellations


# Locate the data of the first subject
sub= np.linspace(1,40,40, dtype=np.int32)
ses=[1]

main_folder= r'D:\ROI Schizo'
func=[]

for subjects in sub:
    subject=str("{:02d}".format(subjects))
    
    for sessions in ses:
        session=str(sessions)
        files1 = sorted(os.listdir( main_folder+'\sub-'+subject +'\\func' ))
        files1=files1[24]
        
        mystring1= main_folder+'\sub-'+subject +'\\func\\'
        filenames1=[ mystring1+files1];

        func.append(filenames1)

    
    

# Use the cortical Harvard-Oxford atlas
atlas_harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
atlas_harvard_oxford.labels=['ROI1','ROI2','ROI3','ROI4','ROI5','ROI6','ROI7','ROI8']

Schizo_Healthy= r'D:\ROI Schizo\ROIs\Schizo-Healthy'

#########################################################################3

#atlas_harvard_oxford.maps= Schizo_Healthy+ '\my_mask_1.ROIs.nii' #Accuracy 52.3%
# Individual accuracy from 1 to 8 is [53, 52, 51.2, 53.77, 53.71, 53.4, 53.3, 49] 

atlas_harvard_oxford.maps= Schizo_Healthy+ '\\roi_1_to_8.nii' # Accuracy 64.45%

#plotting.plot_prob_atlas(atlas_harvard_oxford.maps)

# ROI time series
ts=[]
for bold in func:  
    a=roi.extract_timecourse_from_nii(atlas_harvard_oxford, bold, t_r=2)
    ts.append(a.iloc[:,:].values)


ts_schizo= ts[0:20]
ts_healthy = ts[20:40]

###########################################################################################


schizo_series=list(chain.from_iterable(ts_schizo))
schizo_series=np.array(schizo_series)

healthy_series=list(chain.from_iterable(ts_healthy))
healthy_series=np.array(healthy_series)

###########################################################################################

import nibabel as nb
fname = atlas_harvard_oxford.maps
img = nb.load(fname)
nb.save(img, fname.replace('.img', '.nii'))



mask_parcel=Parcellations('rena', n_parcels=100, random_state=0,mask=atlas_harvard_oxford.maps)

mask_parcel.fit(func[0][0])


kmeans_labels_img = mask_parcel.labels_img_

mean_func_image= 'D:\ROI Schizo\ROIs\Schizo-Healthy\single_subj_T1.nii'
plotting.plot_roi(kmeans_labels_img, mean_func_image,
                  title="KMeans parcellation",
                  display_mode='xz')

kmeans_labels_img.to_filename('roi1_parcellation.nii')





# Import dictionary learning algorithm from decomposition module and call the
# object and fit the model to the functional datasets
from nilearn.decomposition import DictLearning

# Initialize DictLearning object
dict_learn = DictLearning(n_components=2,mask=atlas_harvard_oxford.maps,
                          random_state=0)
# Fit to the data
dict_learn.fit(func[0][0])
# Resting state networks/maps in attribute `components_img_`
# Note that this attribute is implemented from version 0.4.1.
# For older versions, see the note section above for details.
components_img = dict_learn.components_img_

# Visualization of functional networks
# Show networks using plotting utilities
from nilearn import plotting

plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                         title='Dictionary Learning maps')


components_img.to_filename('dict_roi.nii')






