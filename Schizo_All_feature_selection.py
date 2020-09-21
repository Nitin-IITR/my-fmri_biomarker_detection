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


df=series_frame

list_train=list( np.concatenate((np.linspace(0,16*150-1,16*150,dtype=np.int32),np.linspace(20*150,36*150-1,16*150,dtype=np.int32))) )
    
X= df.iloc[list_train,:].values
Y = np.concatenate((np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))))


list_test=list( np.concatenate((np.linspace(16*150,20*150-1,4*150,dtype=np.int32),np.linspace(36*150,40*150-1,4*150,dtype=np.int32))) )    

X_test= df.iloc[list_test,:].values
Y_test = np.concatenate((np.zeros(int(len(X_test)/2)), np.ones(int(len(X_test)/2))))

####################################################################################
df_train1 = df.iloc[list_train,:].reset_index(drop=True)

schizo_series_all222 = df_train1[0:16*150].reset_index(drop=True)
healthy_series_all222= df_train1[16*150:32*150].reset_index(drop=True)



####################################################################################
####################################################################################
####################  Kbest Method #################################

from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

acs_all=[]

for feature_num in range(1,no_of_features):
    
    fs = SelectKBest(score_func=f_regression, k=feature_num)
    # apply feature selection
    X_selected = fs.fit_transform(X, Y)
#    print(X_selected.shape)
    
    ##################   Cross Validation  SVM  ############################################
    
    
    acs=[]
    
    for i in range(1000):
        from sklearn.model_selection import train_test_split
        X_train,X_crossv, Y_train, Y_crossv = train_test_split(X_selected,Y, test_size =0.2)
        
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test= sc_X.transform(X_crossv)
        
        from sklearn.svm import SVC
        svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
        svc_reg.fit(X_train,Y_train)
    
        Y_pred = svc_reg.predict(X_crossv)
    
        
        from sklearn.metrics import confusion_matrix
        cmf= confusion_matrix(Y_crossv,Y_pred)
        
        from sklearn.metrics import accuracy_score
        acs.append(accuracy_score(Y_crossv,Y_pred))
    
    
    acs1 =np.mean(acs) 
    acs1=acs1*100
    
    std=np.std(acs)
    std=std*100
    
    acs_all.append([acs1,std])




####################################################################################
####################################################################################
#########################  Linear SVC  #############################################

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel



all_acc=[]
thresholds_C=[]
no_of_features=[]

for C in np.linspace(0.1,20,1000):

    acs=[]
    
    #    for i in range(500):    
    
    from sklearn.model_selection import train_test_split
    X_train,X_crossv, Y_train, Y_crossv = train_test_split(X,Y, test_size =0.2,random_state=10)
  
    lsvc = LinearSVC(C=C, penalty="l1", dual=False,max_iter=500000,random_state=10).fit(X_train, Y_train)
    model = SelectFromModel(lsvc, prefit=True)
    X_train = model.transform(X_train)
    X_crossv = model.transform(X_crossv)
    ##################   Cross Validation  SVM  ############################################

    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_crossv= sc_X.transform(X_crossv)
    
    from sklearn.svm import SVC
    svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
    svc_reg.fit(X_train,Y_train)

    Y_pred = svc_reg.predict(X_crossv)

    
    from sklearn.metrics import confusion_matrix
    cmf= confusion_matrix(Y_crossv,Y_pred)
    
    from sklearn.metrics import accuracy_score
    acs.append(accuracy_score(Y_crossv,Y_pred))

    
    no_of_features.append(len(X_crossv[0]))
    thresholds_C.append(C)
    acs1 =np.mean(acs) 
    acs1=acs1*100
    
    std=np.std(acs)
    std=std*100
    
    all_acc.append([acs1,std])
    

all_acc1=list(chain.from_iterable(all_acc))
all_acc_iter=np.array(all_acc1)

acs_part=all_acc_iter[list(np.linspace(0,len(all_acc_iter)-2,int(len(all_acc_iter)//2), dtype=np.int32))]
std_part= all_acc_iter[list(np.linspace(1,len(all_acc_iter)-1,int(len(all_acc_iter)//2), dtype=np.int32))]

acs_and_thers= pd.DataFrame({'acs': acs_part,'std': std_part,'C_value': thresholds_C,'features':no_of_features})
acs_sorted= acs_and_thers.sort_values(by=['acs','features'], ascending=[False,True])
acs_sorted= acs_sorted.reset_index()

acs_sorted.to_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\thershold_and_acc_for_schizo_corr_t_test_222Regions_Linear_SVC_feature_selection.csv')



####################################################################################
################# Final Testing SVM   #########################################

all_acc=[]
thresholds_C=[]
no_of_features=[]

X_train=X
Y_train=Y

ff=acs_sorted.iloc[0:164,3].values

for C in ff:
    
    lsvc = LinearSVC(C=C, penalty="l1", dual=False,max_iter=500000,random_state=10).fit(X, Y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    X_test1 = model.transform(X_test)
    ##################   Test Validation  SVM  ############################################
    
    no_of_features.append(len(X_new[0]))
    thresholds_C.append(C)
    
    acs=[]
    
#    for i in range(1):
        
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_new)
    X_test1= sc_X.transform(X_test1)
    
    from sklearn.svm import SVC
    svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
    svc_reg.fit(X_train,Y_train)

    Y_pred = svc_reg.predict(X_test1)

    
    from sklearn.metrics import confusion_matrix
    cmf= confusion_matrix(Y_test,Y_pred)
    
    from sklearn.metrics import accuracy_score
    acs.append(accuracy_score(Y_test,Y_pred))

    
    acs1 =np.mean(acs) 
    acs1=acs1*100
    
    std=np.std(acs)
    std=std*100
    
    all_acc.append([acs1,std])
    


####################################################################################
####################################################################################
####################################################################################
############################# Stepwise Regression ##################################


import pandas as pd
from statsmodels.api import OLS
import statsmodels.api as sm


def stepwise_selection(data, target,SL_in=0.05,SL_out = 0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<SL_in):
            best_features.append(new_pval.idxmin())
            while(len(best_features)>0):
                best_features_with_constant = sm.add_constant(data[best_features])
                p_values = OLS(target, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= SL_out):
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                else:
                    break 
        else:
            break
    return best_features


####################################################################################
###################### Forward Selection ###########################################


all_acc=[]
thresholds_C=[]
no_of_features=[]

for threshold_in in np.linspace(0.001,1,100):

    acs=[]
    
    #    for i in range(500):    
    
    from sklearn.model_selection import train_test_split
    X_train,X_crossv, Y_train, Y_crossv = train_test_split(X,Y, test_size =0.2,random_state=20)
  
    X_indices= stepwise_selection(pd.DataFrame(X_train),Y_train,SL_in=0.1,SL_out = 0.1)


    ##################   Cross Validation  SVM  ############################################

    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train[:, X_indices])
    X_crossv= sc_X.transform(X_crossv[:, X_indices])
    
    from sklearn.svm import SVC
    svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
    svc_reg.fit(X_train,Y_train)

    Y_pred = svc_reg.predict(X_crossv)

    
    from sklearn.metrics import confusion_matrix
    cmf= confusion_matrix(Y_crossv,Y_pred)
    
    from sklearn.metrics import accuracy_score
    acs.append(accuracy_score(Y_crossv,Y_pred))

    
    no_of_features.append(len(X_crossv[0]))
    thresholds_C.append(threshold_in)
    acs1 =np.mean(acs) 
    acs1=acs1*100
    
    std=np.std(acs)
    std=std*100
    
    all_acc.append([acs1,std])
   

all_acc1=list(chain.from_iterable(all_acc))
all_acc_iter=np.array(all_acc1)

acs_part=all_acc_iter[list(np.linspace(0,len(all_acc_iter)-2,int(len(all_acc_iter)//2), dtype=np.int32))]
std_part= all_acc_iter[list(np.linspace(1,len(all_acc_iter)-1,int(len(all_acc_iter)//2), dtype=np.int32))]

acs_and_thers= pd.DataFrame({'acs': acs_part,'std': std_part,'C_value': thresholds_C,'features':no_of_features})
acs_sorted= acs_and_thers.sort_values(by=['acs','features'], ascending=[False,True])
acs_sorted= acs_sorted.reset_index()

#acs_sorted.to_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\thershold_and_acc_for_schizo_corr_t_test_222Regions_Linear_SVC_feature_selection.csv')



####################################################################################
###################### Backward Elimnation ###########################################

all_acc=[]
thresholds_C=[]
no_of_features=[]

for threshold_in in np.linspace(0.001,1,500):

    acs=[]
    
    #    for i in range(500):    
    
    from sklearn.model_selection import train_test_split
    X_train,X_crossv, Y_train, Y_crossv = train_test_split(X,Y, test_size =0.2,random_state=10)
  
    X_indices= backward_regression(pd.DataFrame(X_train),Y_train,threshold_in=threshold_in)



    ##################   Cross Validation  SVM  ############################################

    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train[:, X_indices])
    X_crossv= sc_X.transform(X_crossv[:, X_indices])
    
    from sklearn.svm import SVC
    svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
    svc_reg.fit(X_train,Y_train)

    Y_pred = svc_reg.predict(X_crossv)

    
    from sklearn.metrics import confusion_matrix
    cmf= confusion_matrix(Y_crossv,Y_pred)
    
    from sklearn.metrics import accuracy_score
    acs.append(accuracy_score(Y_crossv,Y_pred))

    
    no_of_features.append(len(X_crossv[0]))
    thresholds_C.append(C)
    acs1 =np.mean(acs) 
    acs1=acs1*100
    
    std=np.std(acs)
    std=std*100
    
    all_acc.append([acs1,std])
    

all_acc1=list(chain.from_iterable(all_acc))
all_acc_iter=np.array(all_acc1)

acs_part=all_acc_iter[list(np.linspace(0,len(all_acc_iter)-2,int(len(all_acc_iter)//2), dtype=np.int32))]
std_part= all_acc_iter[list(np.linspace(1,len(all_acc_iter)-1,int(len(all_acc_iter)//2), dtype=np.int32))]

acs_and_thers= pd.DataFrame({'acs': acs_part,'std': std_part,'C_value': thresholds_C,'features':no_of_features})
acs_sorted= acs_and_thers.sort_values(by=['acs','features'], ascending=[False,True])
acs_sorted= acs_sorted.reset_index()

#acs_sorted.to_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\thershold_and_acc_for_schizo_corr_t_test_222Regions_Linear_SVC_feature_selection.csv')



####################################################################################
####################################################################################
####################################################################################
############################# Forward and Backward Feature Selection  ##############

# forward=True,floating=False   :  This is for "Forward Selection"
# forward=False,floating=False  :  This is for "Backward Selection" 
# forward=True,floating=True    :  This is for "Stepwise Regression" 

####################################################################################

#importing the necessary libraries
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import SVC

all_acc=[]
thresholds_C=[]
no_of_features=[]

for n_features in [40,60,80,100]:
    
   #####################################################################################
   ###################### Data Separatio to Train and Cross Validation     #############
   ######################   ######################      ###################### 
    
    ############################################
    import random
    
    schizo_series_test =pd.DataFrame([])
    randomlist = random.sample(range(0,15),4)
    
    for i in range(0,4):
    
        n=randomlist[i]
        schizo_series_test=pd.concat([schizo_series_test,schizo_series_all222[150*n:150*(n+1)] ])
    print(randomlist)
    
    index_drop = list(schizo_series_test.index)
    schizo_series_train=schizo_series_all222.drop(index_drop)
    schizo_series_train=schizo_series_train.reset_index(drop=True)
    
    ################## HEALTHY DATA PREP ######################################
    
    import random
    
    healthy_series_test =pd.DataFrame([])
    randomlist = random.sample(range(0,15),4)    
    
    for i in range(0,4):
    
        n=randomlist[i]
        healthy_series_test=pd.concat([healthy_series_test,healthy_series_all222[150*n:150*(n+1)] ])
    print(randomlist)
    
    index_drop = list(healthy_series_test.index)
    healthy_series_train=healthy_series_all222.drop(index_drop)
    healthy_series_train=healthy_series_train.reset_index(drop=True)
    
    #############################################################################
    #############################################################################
    
    ####### TRAIN DATA #############################
    
    df_trainX = schizo_series_train.append(healthy_series_train)
    df_trainX=df_trainX.reset_index(drop=True)
    df_trainY = pd.DataFrame(np.concatenate((np.zeros(len(schizo_series_train)), np.ones(len(healthy_series_train)))))
    
    
    df_train = pd.concat([df_trainX, df_trainY], axis=1, sort=False)
    
    ####### TEST DATA #############################
    
    df_testX = schizo_series_test.append(healthy_series_test)
    df_testX=df_testX.reset_index(drop=True)
    df_testY = pd.DataFrame(np.concatenate((np.zeros(len(schizo_series_test)), np.ones(len(healthy_series_test)))))
    
    df_test = pd.concat([df_testX, df_testY], axis=1, sort=False)
 
    ############################################################
        
#    df_train = df_train.sample(frac=1).reset_index(drop=True)
    X_train= df_train.iloc[:,0:len(df_train.columns)-1].values
    Y_train=df_train.iloc[:,len(df_train.columns)-1].values
    
#    df_test = df_test.sample(frac=1).reset_index(drop=True)
    X_crossv= df_test.iloc[:,0:len(df_test.columns)-1].values
    Y_test=df_test.iloc[:,len(df_test.columns)-1].values
    
    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_crossv= sc_X.transform(X_crossv)
    
    ######################   ######################      ###################### 
    ######################   ######################      ###################### 
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    # Sequential Forward Selection(sfs)
    sfs = SFS(SVC(kernel ='rbf',C=1,gamma='auto'),
               k_features=n_features,                # no. of features to select
               forward=True,floating=True, # This is for "Forward Selection",  For "Backward Selection" Make both of them False, For "Stepwise Regression" make them both False
               scoring = 'accuracy',            # for classification, it can be accuracy, precision, recall, f1-score, etc.
               cv = 0)                          # cv is k fold cross validation



#    s=sfs.subsets_
    ##################   Cross Validation  SVM  ############################################

    sfs.fit(X_train, Y_train)
    X_indices= list(sfs.k_feature_names_)   # to get the final set of features
    
    X_indices1=[]
    for j in range(len(X_indices)):
        X_indices1.append(int(X_indices[j]))
    
    
    ############################################
    X_train= X_train[:, X_indices1]
    X_crossv= X_crossv[:, X_indices1]
    
    ############################################
       
    
    from sklearn.svm import SVC
    svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
    svc_reg.fit(X_train,Y_train.ravel())
    
    ####################### Testing data ###################
    df_train2=df_train1.iloc[:, X_indices1]
    
    total_subjects=32
    Y_avg_predict=[]
    Y_avg_true= np.concatenate((np.zeros(int(total_subjects//2)), np.ones(int(total_subjects//2))))
    
    for total_sub in range(total_subjects):
        
        Y_pred = svc_reg.predict(df_train2[150*total_sub : 150*(total_sub+1)].iloc[:,:].values)
        Y_avg_predict.append(np.mean(Y_pred))
    
    
    
    for h in range(len(Y_avg_predict)):
        if Y_avg_predict[h]<0.5:
            Y_avg_predict[h]=0
        
        else:
            Y_avg_predict[h]=1
            
            
    from sklearn.metrics import confusion_matrix
    cmf= confusion_matrix(Y_avg_predict,Y_avg_true)

    from sklearn.metrics import accuracy_score
    scores_avg = accuracy_score(Y_avg_predict,Y_avg_true) 
   

    all_acc.append(scores_avg)
    thresholds_C.append(X_indices1)
    no_of_features.append(n_features)
    
    


    
 

acs_part=all_acc
acs_and_thers= pd.DataFrame({'acs': acs_part,'X_index': thresholds_C,'features':no_of_features})
acs_sorted= acs_and_thers.sort_values(by=['acs','features'], ascending=[False,True])
acs_sorted= acs_sorted.reset_index()

acs_sorted.to_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\n_features_and_acc_for_schizo_series_222Regions_Stepwise_feature_selection_mlxtend.csv')



###########################################################################################

# Plotting number of feature v/s accuracy
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()
###########################################################################################
###########################################################################################
##########



#################################################################################

all_acc=[]
thresholds_C=[]
no_of_features=[]

X_train=X
Y_train=Y

ff= list(acs_sorted.iloc[0:64,4].values)

for n_features in ff:


    # Sequential Forward Selection(sfs)
    sfs = SFS(SVC(kernel ='rbf',C=1,gamma='auto'),
               k_features=49,                # no. of features to select
               forward=True,floating=True, # This is for "Forward Selection",  For "Backward Selection" Make both of them False, For "Stepwise Regression" make them both False
               scoring = 'accuracy',            # for classification, it can be accuracy, precision, recall, f1-score, etc.
               cv = 0)                          # cv is k fold cross validation



#    s=sfs.subsets_
    ##################   Cross Validation  SVM  ############################################

    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X)
    X_test1= sc_X.transform(X_test)
    
    
    sfs.fit(X_train, Y_train)
    X_indices= list(sfs.k_feature_names_)   # to get the final set of features
    
    X_indices1=[]
    for j in range(len(X_indices)):
        X_indices1.append(int(X_indices[j]))
    
    
    ############################################
    X_train= X_train[:, X_indices1]
    X_test1= X_test1[:, X_indices1]
    
    
#    for i in range(1):
        
    acs=[]
    
    from sklearn.svm import SVC
    svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
    svc_reg.fit(X_train,Y_train)

    Y_pred = svc_reg.predict(X_test1)

    
    from sklearn.metrics import confusion_matrix
    cmf= confusion_matrix(Y_test,Y_pred)
    
    from sklearn.metrics import accuracy_score
    acs.append(accuracy_score(Y_test,Y_pred))

    
    acs1 =np.mean(acs) 
    acs1=acs1*100
    
    std=np.std(acs)
    std=std*100
    
    no_of_features.append(n_features)
    thresholds_C.append(X_indices1)
    acs1 =np.mean(acs) 
    acs1=acs1*100
    
    std=np.std(acs)
    std=std*100
    
    all_acc.append([acs1,std])
    
    
 
