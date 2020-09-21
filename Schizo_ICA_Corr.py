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


# Locate the data of the first subject
number_of_sub= 90
sub= np.linspace(1,number_of_sub,number_of_sub, dtype=np.int32)
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

func1=[]
for i in range(len(func)):
    func1.append(func[i][0])
    
func=func1

#####################################################################

# Use the cortical Harvard-Oxford atlas
atlas_harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
atlas_harvard_oxford.labels=['ROI1']


#########################################################################3
############### DIFFERENT ICA MAPS     #################
ICA_folder= r'D:\ROI Schizo\ICA'

# Accuracy Series: 69.45% || series+ 8roi:74.85% || Corr: 71%
#atlas_harvard_oxford.maps= ICA_folder+ '\\ICA_schizo_2_comp.nii'  

# Accuracy Series: 92% || series+ 8roi:92.5%% || Corr: 98% ||104 Features
#atlas_harvard_oxford.maps= ICA_folder+ '\\ICA_schizo_25_comp.nii' 

############################## Mostly used ##############################
# Accuracy Series:Thershold=0.5 92% || series+ 8roi: 92.5%% || Corr: 98% || 123 Features
# Accuracy Series:Thershold=1 95.8% || series+ 8roi:  || Corr:  || 195 Features
# Accuracy Series:Thershold=2 96% || series+ 8roi:  || Corr: 63% || 222 Features
#atlas_harvard_oxford.maps= ICA_folder+ '\\ICASSO25_schizo_25_comp.nii'


# 
atlas_harvard_oxford.maps= ICA_folder+ '\\ICASSO25_schizo_25_comp_90sub.nii'


##########################################################################################

# Accuracy Series:Thershold=2 % || series+ 8roi:  || Corr: % || 222 Features
#atlas_harvard_oxford.maps= ICA_folder+ '\\ICASSO25_schizo_12_comp.nii'


# Accuracy Series: 84.6% || series+ 8roi: 84.63%% || Corr: 68.16%
#atlas_harvard_oxford.maps= ICA_folder+ '\\ICASSO25_schizo_23_comp.nii'

# Accuracy Series: 89.4% || series+ 8roi: 90.1% || Corr: 66.15%  || 84 Features
#atlas_harvard_oxford.maps= ICA_folder+ '\\ICASSO30_schizo_30_comp.nii'

# Accuracy Series: 88.13% || series+ 8roi: 88.9.1% || Corr: 70.5% 
#atlas_harvard_oxford.maps= ICA_folder+ '\\ICASSO35_schizo_35_comp.nii'

# Accuracy Series:Thershold=0.1:  62.21% || series+ 8roi:  || Corr: % || 16 Features
# Group SVM: 68.2%
# Accuracy of each 16 egion extractor=[50,52,51.3,52.4,49.7,50.6,50.3,49.7,51,50.4,48.9,48.8,51.9,51.1]
# Accuracy using EEG type band extraction and then feature ext or directly feature ext is below 50%
#atlas_harvard_oxford.maps= ICA_folder+ '\\ICASSO25_11_17.nii'


plotting.plot_prob_atlas(atlas_harvard_oxford.maps)

###################################################################################
###################################################################################
##################### Dictionary Learning ########################################
'''
# Import dictionary learning algorithm from decomposition module and call the
# object and fit the model to the functional datasets
from nilearn.decomposition import DictLearning

# Initialize DictLearning object
dict_learn = DictLearning(n_components=25, smoothing_fwhm=None,
                          memory=r'D:\ROI Schizo\ICA\Dict_learning\Nilearn_cache', 
                          memory_level=5, random_state=0, n_jobs=6)
# Fit to the data
dict_learn.fit(func)
# Resting state networks/maps in attribute `components_img_`
# Note that this attribute is implemented from version 0.4.1.
# For older versions, see the note section above for details.
components_img = dict_learn.components_img_

# Visualization of functional networks
# Show networks using plotting utilities
from nilearn import plotting

plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                         title='Dictionary Learning maps')


from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show

for i, cur_img in enumerate(iter_img(components_img)):
    plot_stat_map(cur_img, display_mode="z", title="Comp %d" % i,
                  cut_coords=1, colorbar=False)

'''
###################################################################################
###################################################################################
###################################################################################
####################  PARCELLATION METHOD ##########################################



from nilearn.regions import Parcellations
# Agglomerative Clustering: ward

# We build parameters of our own for this object. Parameters related to
# masking, caching and defining number of clusters and specific parcellations


ward = Parcellations(method='ward', n_parcels=1000,
                     standardize=False, smoothing_fwhm=None,
                     memory=r'D:\ROI Schizo\Nilearn_parcell\Nilearn_parcell_cache', memory_level=1)
# Call fit on functional dataset: single subject (less samples).


for bold in func:
    ward.fit(bold)
    print('bold is fitted')



ward_labels_img = ward.labels_img_
ward_mask = ward.masker_
# Now, ward_labels_img are Nifti1Image object, it can be saved to file
# with the following code:
ward_labels_img.to_filename(r'D:\ROI Schizo\Nilearn_parcell\schizo_90sub_1000_ward_parcel.nii')



from nilearn import plotting
from nilearn.image import mean_img, index_img

first_plot = plotting.plot_roi(ward_labels_img, title="Ward parcellation",
                               display_mode='xz')

# Grab cut coordinates from this plot to use as a common for all plots
cut_coords = first_plot.cut_coords

coordinates=nilearn.plotting.find_parcellation_cut_coords(ward_labels_img,return_label_names=True)




###################################################################################
###################################################################################
ward_labels_img= r'D:\ROI Schizo\Nilearn_parcell\schizo_90sub_1000_ward_parcel.nii'



    #     Signal extraction from parcellations
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# ConenctivityMeasure from Nilearn uses simple 'correlation' to compute
# connectivity matrices for all subjects in a list
connectome_measure = ConnectivityMeasure(kind='correlation')

# useful for plotting connectivity interactions on glass brain
from nilearn import plotting

# create masker to extract functional data within atlas parcels
masker = NiftiLabelsMasker(labels_img=ward_labels_img, standardize=True)

# extract time series from all subjects and concatenate them
time_series = []
for bold in func:
    time_series.append(masker.fit_transform(bold))

# calculate correlation matrices across subjects and display
correlations = list(connectome_measure.fit_transform(time_series))


################# Saving time_series and correlation matrices to hard disk ####

time_seriespp=list(chain.from_iterable(time_series))
time_seriespp=np.array(time_seriespp)

time_series_frame= pd.DataFrame(time_seriespp)
time_series_frame.to_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\schizo_time_series_1000_ward_parcellation.csv',header=False, index=False)
########################################

correlationspp=list(chain.from_iterable(correlations))
correlationspp=np.array(correlationspp)

correlations_frame= pd.DataFrame(correlationspp)
correlations_frame.to_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\schizo_correlation_1000_ward_parcellation.csv',header=False, index=False)














###################################################################################
###################################################################################
###################################################################################
###################################################################################


#################################################################################
#######################  REGION EXTRACTOR  ######################################

################################################################################
# Extract regions from networks
# ------------------------------


# Import Region Extractor algorithm from regions module
# threshold=0.5 indicates that we keep nominal of amount nonzero voxels across all
# maps, less the threshold means that more intense non-voxels will be survived.
from nilearn.regions import RegionExtractor

extractor = RegionExtractor(atlas_harvard_oxford.maps, threshold=2,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True, min_region_size=512)
# Just call fit() to process for regions extraction
extractor.fit()
# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Each region index is stored in index_
regions_index = extractor.index_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         '\nEach separate color of region indicates extracted region'
         % (n_regions_extracted, 3))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title)


#regions_extracted_img.to_filename('region_ext_img_ICASSO25_schizo_25_comp_222features.nii')



#######################################################################################


################################################################################
# Compute correlation coefficients
# ---------------------------------

# First we need to do subjects timeseries signals extraction and then estimating
# correlation matrices on those signals.
# To extract timeseries signals, we call transform() from RegionExtractor object
# onto each subject functional data stored in func_filenames.
# To estimate correlation matrices we import connectome utilities from nilearn
from nilearn.connectome import ConnectivityMeasure

time_series=[]
correlations = []
# Initializing ConnectivityMeasure object with kind='correlation'
connectome_measure = ConnectivityMeasure(kind='correlation')
for bold in func:
    # call transform from RegionExtractor object to extract timeseries signals
    timeseries_each_subject = extractor.transform(bold)
    
    # append time series
    time_series.append(timeseries_each_subject)
    
    # call fit_transform from ConnectivityMeasure object
    correlation = connectome_measure.fit_transform([timeseries_each_subject])
    # saving each subject correlation to correlations
    correlations.append(np.reshape(correlation,(n_regions_extracted,n_regions_extracted)))

# Mean of all correlations
import numpy as np
mean_correlations11 = np.mean(correlations, axis=0).reshape(n_regions_extracted,
                                                          n_regions_extracted)

################################################

################# Saving time_series and correlation matrices to hard disk ####

time_seriespp=list(chain.from_iterable(time_series))
time_seriespp=np.array(time_seriespp)

time_series_frame= pd.DataFrame(time_seriespp)
time_series_frame.to_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\schizo_time_series_ICASSO25_schizo_25_comp_90sub_thresh2_min_region_size_512.csv',header=False, index=False)
########################################

correlationspp=list(chain.from_iterable(correlations))
correlationspp=np.array(correlationspp)

correlations_frame= pd.DataFrame(correlationspp)
correlations_frame.to_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\schizo_correlations_ICASSO25_schizo_25_comp_90sub_thresh2_min_region_size_512.csv',header=False, index=False)


###############################################################################
######################### TIME SERIES ###################################
# To directly import correlation coefficients

series_frame= pd.read_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\schizo_time_series_ICASSO25_schizo_25_comp_90sub_thresh2_min_region_size_512.csv',
                          header=None)

series_array= series_frame.iloc[:,:].values

time_series=  list(series_array.reshape(90,150,172))
####################################################################################
####################################################################################

ts_schizo= time_series[0:20]
ts_healthy = time_series[20:40]

schizo_series_allx=list(chain.from_iterable(ts_schizo))
schizo_series_all16=np.array(schizo_series_allx)

healthy_series_allx=list(chain.from_iterable(ts_healthy))
healthy_series_all16=np.array(healthy_series_allx)

####################################################################################
###################### CORRELATION #############################################

# To directly import correlation coefficients
correlations_frame= pd.read_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\schizo_correlations_ICASSO25_schizo_25_comp_90sub_thresh2_min_region_size_512.csv',
                          header=None)

correlations_array= correlations_frame.iloc[:,:].values

correlations=  list(correlations_array.reshape(90,172,172))
####################################################################################
####################################################################################




corr_schizo= correlations[0:20]
corr_healthy = correlations[20:40]

schizo_corr_allx=list(chain.from_iterable(corr_schizo))
schizo_corr_all16=np.array(schizo_corr_allx)

healthy_corr_allx=list(chain.from_iterable(corr_healthy))
healthy_corr_all16=np.array(healthy_corr_allx)

##################### Mean correlation ########################################
#Accuracy :
# Using ICASSO25_schizo_25_comp.nii at 0.5 region ext thershlod:
#    Using t_value > 5 : acc= 85%: 3 features, > 4.5: acc=89.5%: 9 features, > 4.2: acc=95%: 18 features
#    > 3.5: acc=97.2%: 51 features, > 3.5: acc=98%: 90 features, > 3.1: acc=97%: 108 features
#    > 3: acc=98.62%: 131 features, > 2.5: acc=97%: 340 features and then decreases
#    > 3.4 acc=99% 60 features and if we search nearby the no. of deatures remain same so does the accuracy

#Using ICASSO25_schizo_25_comp.nii at 0.5 region ext thershlod:222 regions
#  Using t_value > 3.78 : acc= 100%: 47 features ad more details in excel file



schizo_corr_3D= np.array([[[]]])

schizo_corr_3D= np.array(correlations[0:20])
healthy_corr_3D= np.array(correlations[20:40])

corr_t_test= sp.stats.ttest_ind(schizo_corr_3D,healthy_corr_3D,axis=0)
Corr_t_stats= corr_t_test[0]
Corr_p_stats= corr_t_test[1]


corr_t_frame=pd.DataFrame(columns=['t_value','i_value','j_value'])
corr_p_frame=pd.DataFrame(columns=['p_value','i_value','j_value'])

for i in range(len(Corr_t_stats)):
    for j in range(len(Corr_t_stats)):
        
        if abs(Corr_t_stats[i][j]) >4.54309  :
            tem_frame1=pd.DataFrame({'t_value':[Corr_t_stats[i][j]],'i_value':[i],'j_value':[j]})
            corr_t_frame=corr_t_frame.append(tem_frame1)
            
        if abs(Corr_p_stats[i][j]) <10**(-4.8) :
            tem_frame2=pd.DataFrame({'p_value':[Corr_p_stats[i][j]],'i_value':[i],'j_value':[j]})
            corr_p_frame=corr_p_frame.append(tem_frame2)            
                        
corr_p_frame=corr_p_frame.reset_index(drop=True)            
corr_t_frame=corr_t_frame.reset_index(drop=True)

i_value= corr_t_frame[0:(len(corr_t_frame)//2)]['i_value']
j_value= corr_t_frame[0:(len(corr_t_frame)//2)]['j_value']

len(i_value)

corr_schizo= correlations[0:20]
corr_healthy = correlations[20:40]
corr_all= corr_schizo+corr_healthy

main_corr_list=[]

for i in range(len(corr_all)):
    corr_one_person=corr_all[i]
    a=[]
    for l,k in zip(i_value,j_value):
        a.append(corr_one_person[l,k])
        
    main_corr_list.append(a)
    
main_corr_frame=pd.DataFrame(main_corr_list)







# Anser in format t-value, p-value: and more difference means greaater t-value and smaller p-value
sp.stats.ttest_ind([1,2,3,4,5,6],[1,2,3,4,5,6])
sp.stats.ttest_ind([1,2,3,4,5,6],[7,8,9,10,11,12])
sp.stats.ttest_ind([1,1,1,1,1,1],[1,1,1,1,1,1])

########################################################################
########################################################################
### Preparing new testing data






# Locate the data of the first subject
sub= np.linspace(1,10,10, dtype=np.int32)
ses=[1]

main_folder= r'D:\Schizo Testing'
func_test=[]

for subjects in sub:
    subject=str("{:02d}".format(subjects))
    
    for sessions in ses:
        session=str(sessions)
        files1 = sorted(os.listdir( main_folder+'\sub-'+subject +'\\func' ))
        files1=files1[24]
        
        mystring1= main_folder+'\sub-'+subject +'\\func\\'
        filenames1=[ mystring1+files1];

        func_test.append(filenames1)

func_test1=[]
for i in range(len(func_test)):
    func_test1.append(func_test[i][0])
    
func_test=func_test1


###########################################

time_series_test=[]
correlations_test = []
# Initializing ConnectivityMeasure object with kind='correlation'
connectome_measure = ConnectivityMeasure(kind='correlation')
for bold in func_test:
    # call transform from RegionExtractor object to extract timeseries signals
    timeseries_each_subject = extractor.transform(bold)
    
    # append time series
    time_series_test.append(timeseries_each_subject)
    
    # call fit_transform from ConnectivityMeasure object
    correlation = connectome_measure.fit_transform([timeseries_each_subject])
    # saving each subject correlation to correlations
    correlations_test.append(np.reshape(correlation,(n_regions_extracted,n_regions_extracted)))

# Mean of all correlations
import numpy as np
mean_correlations_test11 = np.mean(correlations_test, axis=0).reshape(n_regions_extracted,
                                                          n_regions_extracted)


###########################################

main_corr_list_test=[]

for i in range(len(correlations_test)):
    corr_one_person_test=correlations_test[i]
    a=[]
    for l,k in zip(i_value,j_value):
        a.append(corr_one_person_test[l,k])
        
    main_corr_list_test.append(a)
    
main_corr_frame_test=pd.DataFrame(main_corr_list_test)



#######################################################################
    



acs=[]

for i in range(10):
    
    
####################################################################### 
    ###     PREPARING TEST DATA
    
    df_testY = pd.DataFrame(np.concatenate((np.zeros(int(len(main_corr_frame_test)//2)), np.ones(int(len(main_corr_frame_test)//2)))))
    df_testX= main_corr_frame_test
    df_test = pd.concat([df_testX, df_testY], axis=1, sort=False)
    
    ######
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    X_test= df_test.iloc[:,0:len(df_test.columns)-1].values
    Y_test=df_test.iloc[:,len(df_test.columns)-1].values
    
    ##### ###################################  ###################
    ###     PREPARING TRAIN DATA
    df_trainY = pd.DataFrame(np.concatenate((np.zeros(int(len(main_corr_frame)//2)), np.ones(int(len(main_corr_frame)//2)))))
    df_trainX= main_corr_frame
    df_train = pd.concat([df_trainX, df_trainY], axis=1, sort=False)
    
    ######
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    X_train= df_train.iloc[:,0:len(df_train.columns)-1].values
    Y_train=df_train.iloc[:,len(df_train.columns)-1].values
    
    
    #####################################
    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test= sc_X.transform(X_test)
        
    
#######################################################################
    
    from sklearn.svm import SVC
    svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
    svc_reg.fit(X_train,Y_train)

    Y_pred = svc_reg.predict(X_test)

    
    from sklearn.metrics import confusion_matrix
    cmf= confusion_matrix(Y_test,Y_pred)
    
    from sklearn.metrics import accuracy_score
    acs.append(accuracy_score(Y_test,Y_pred))


acs1 =np.mean(acs) 
acs1=acs1*100

std=np.std(acs)
std=std*100

acs1











###############################################################################
###############################################################################
################  EEG TYPE FEATURE EXTRACTION (NO GOOD RESULTS)  ##############


schizo_series_all222=pd.DataFrame( schizo_series_all16)
healthy_series_all222=pd.DataFrame(healthy_series_all16)

com_frame = schizo_series_all222.append(healthy_series_all222)

com_frame= com_frame.reset_index(drop=True)

############################################
points_at_time=150


def feature_extraction(method):
    a=[]
    for i in range(n_regions_extracted):
        b=[]
        for j in range(int(len(com_frame)/points_at_time)):  # total length devided by 25(the value used below)
            b.append(method(com_frame.iloc[:,i].values[points_at_time*j:(points_at_time*(j+1))]))
        a.append(b)

    c=pd.concat([pd.DataFrame(a[i]) for i in range(n_regions_extracted)],ignore_index=True,axis=1) 
    
    return c

############################################
a=list()

# Integration
# AB_inte has first 9 columns of all gamma from 9 channels and then of beta etc.
AB_inte = feature_extraction(sp.integrate.simps)

# Approximate entropy
from entropy import app_entropy
AB_app_ent = feature_extraction(app_entropy)


# Sample entropy
import nolds
AB_sam_ent = feature_extraction(nolds.sampen)

# Iqr
AB_iqr = feature_extraction(sp.stats.iqr)

# Mode
AB_mode = feature_extraction(sp.stats.mode)
AB_mode= AB_mode.iloc[:,np.linspace(0,88,45)].values

AB_mode= AB_mode.astype(np.float) 

AB_mode= pd.DataFrame(AB_mode)


# Mean
import statistics
AB_mean = feature_extraction(statistics.mean)

# Std
from astropy.stats import mad_std
AB_std = feature_extraction(mad_std)

#####################################################



























################### TIME SERIES #####################################

    
ts_schizo= time_series[0:20]
ts_healthy = time_series[20:40]

schizo_series_allx=list(chain.from_iterable(ts_schizo))
schizo_series_all16=np.array(schizo_series_allx)

healthy_series_allx=list(chain.from_iterable(ts_healthy))
healthy_series_all16=np.array(healthy_series_allx)

'''########################################
s1= pd.DataFrame(schizo_series_all15)
h1= pd.DataFrame(healthy_series_all15)

all1= s1.append(h1)



################# TRAIN #######################
Y_train=np.concatenate((np.zeros(len(s1)), np.ones(len(h1))))

X_train= all1.iloc[:,:].values

################# TEST #######################
Y_test=np.concatenate((np.zeros(len(s1)), np.ones(len(h1))))

X_test= all1.iloc[:,:].values
'''
###########################################################
###########################################################
###########################################################

s1= pd.DataFrame(schizo_series_all15)
h1= pd.DataFrame(healthy_series_all15)

all1= s1.append(h1)
all1=all1.reset_index(drop=True)
Y=pd.DataFrame(np.concatenate((np.zeros(len(s1)), np.ones(len(h1)))))
all2=pd.concat([all1, Y], axis=1, sort=False)

all3 = all2.sample(frac=1).reset_index(drop=True)


################# TRAIN #######################
Y_train=all3.iloc[:,len(all3.columns)-1].values

X_train= all3.iloc[:,0:len(all3.columns)-1].values

################# TEST #######################
Y_test=all3.iloc[:,len(all3.columns)-1].values

X_test= all3.iloc[:,0:len(all3.columns)-1].values





####################################################################################
###################### CORRELATION #############################################

corr_schizo= correlations[0:20]
corr_healthy = correlations[20:40]

schizo_corr_allx=list(chain.from_iterable(corr_schizo))
schizo_corr_all16=np.array(schizo_corr_allx)

healthy_corr_allx=list(chain.from_iterable(corr_healthy))
healthy_corr_all16=np.array(healthy_corr_allx)











###############################################################################
####################################################################################
##########  PCA  #######################################

## LOW ACCURACY: ONLY 63.11%

from sklearn.decomposition import PCA
pca = PCA(n_components='mle',svd_solver= 'full')
schizo_all_pca=pca.fit_transform(schizo_series_all22)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

#######################################


healthy_all_pca=pca.fit_transform(healthy_series_all22)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

####################################################################################
####################################################################################


    
#schizo_series_all11=schizo_series_all
#healthy_series_all11=healthy_series_all
        
plt.plot(healthy_series_all22[1])        
plt.plot(schizo_series_all22[1])     


# BELOW ONE WILL MAKE ACCURACY WORSE BECAUSE IT GAVE RISE TO OUTLIERS AND AS MANY NUMBERS ARE IN BETWEEN 
# 0 AND 1, SO IT GONNA MAKE BOTH CLASSES CLOSE TO ZERO AND HARDER TO SEPARATE
plt.plot( np.power(healthy_series_all,3))   
plt.plot( np.power(schizo_series_all,3)) 
        
        


















###############################################################################
# Plot resulting connectomes
# ----------------------------

title = 'Correlation between %d regions' % n_regions_extracted

# First plot the matrix
display = plotting.plot_matrix(mean_correlations, vmax=1, vmin=-1,
                               colorbar=True, title=title)

# Then find the center of the regions and plot a connectome
regions_img = regions_extracted_img
coords_connectome = plotting.find_probabilistic_atlas_cut_coords(regions_img)

plotting.plot_connectome(mean_correlations, coords_connectome,
                         edge_threshold='90%', title=title)

################################################################################
# Plot regions extracted for only one specific network
# ----------------------------------------------------
components_img=atlas_harvard_oxford.maps
# First, we plot a network of index=4 without region extraction (left plot)
from nilearn import image

img = image.index_img(components_img, 1)
coords = plotting.find_xyz_cut_coords(img)
display = plotting.plot_stat_map(img, cut_coords=coords, colorbar=False,
                                 title='Showing one specific network')

################################################################################
# Now, we plot (right side) same network after region extraction to show that
# connected regions are nicely seperated.
# Each brain extracted region is identified as separate color.

# For this, we take the indices of the all regions extracted related to original
# network given as 4.
regions_indices_of_map3 = np.where(np.array(regions_index) == 1)

display = plotting.plot_anat(cut_coords=coords,
                             title='Regions from this network')

# Add as an overlay all the regions of index 4
colors = 'rgbcmyk'
for each_index_of_map3, color in zip(regions_indices_of_map3[0], colors):
    display.add_overlay(image.index_img(regions_extracted_img, each_index_of_map3),
                        cmap=plotting.cm.alpha_cmap(color))

plotting.show()




















































