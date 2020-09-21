###############################################################################
###############################################################################
####################### P-value ########################################

################################################################################
################################################################################
random.seed(2)
schizo_test_index=random.sample(range(0,45),12)
random.seed(3)
healthy_test_index=random.sample(range(0,45),12) # 0 to 45 not 45 to 90 because enumerate index always start from 0

list_for_0_to_44= list(np.linspace(0,44,45,dtype=np.int32))
list_for_45_to_89= list(np.linspace(0,44,45,dtype=np.int32)) # 0 to 45 not 45 to 90 because enumerate index always start from 0


schizo_train_index = [j for i, j in enumerate(list_for_0_to_44) if i not in schizo_test_index]
healthy_train_index = [j for i, j in enumerate(list_for_45_to_89) if i not in healthy_test_index]

################################################################################
################################################################################
# Test data
schizo_corr_test= [j for i, j in enumerate(correlations[0:45]) if i not in schizo_train_index]
healthy_corr_test= [j for i, j in enumerate(correlations[45:90]) if i not in healthy_train_index]
correlations_test=schizo_corr_test+healthy_corr_test
############################################
# Train data
corr_schizo= [j for i, j in enumerate(correlations[0:45]) if i not in schizo_test_index]
corr_healthy= [j for i, j in enumerate(correlations[45:90]) if i not in healthy_test_index]
corr_all= corr_schizo+corr_healthy


corr_t_test= sp.stats.ttest_ind(np.array(corr_schizo),np.array(corr_healthy),axis=0)
Corr_t_stats= corr_t_test[0]
Corr_p_stats= corr_t_test[1]



corr_t_frame=pd.DataFrame(columns=['t_value','i_value','j_value'])
corr_p_frame=pd.DataFrame(columns=['p_value','i_value','j_value'])

for i in range(len(Corr_t_stats)):
    for j in range(i+1,len(Corr_t_stats)):
        
        if abs(Corr_t_stats[i][j]) > 2 :
            tem_frame1=pd.DataFrame({'t_value':[Corr_t_stats[i][j]],'i_value':[i],'j_value':[j]})
            corr_t_frame=corr_t_frame.append(tem_frame1)
  
corr_t_frame=corr_t_frame.reset_index(drop=True)

i_value= corr_t_frame['i_value']
j_value= corr_t_frame['j_value']

len(i_value)
    
################### %%%%%%%
    

main_corr_list=[]

for i in range(len(corr_all)):
    corr_one_person=corr_all[i]
    a=[]
    for l,k in zip(i_value,j_value):
        a.append(corr_one_person[l,k])
        
    main_corr_list.append(a)
    
main_corr_frame=pd.DataFrame(main_corr_list)



    
####################################################################################

df=main_corr_frame
X= df.iloc[:,:].values
Y = np.concatenate((np.zeros(len(corr_schizo)), np.ones(len(corr_healthy))))


##################################################################
################################################################## 

main_corr_list_test=[]

for i in range(len(correlations_test)):
    corr_one_person_test=correlations_test[i]
    a=[]
    for l,k in zip(i_value,j_value):
        a.append(corr_one_person_test[l,k])
        
    main_corr_list_test.append(a)
    
#####################            
main_corr_frame_test=pd.DataFrame(main_corr_list_test)

X_test = main_corr_frame_test.iloc[:,:].values
Y_test = np.concatenate((np.zeros(len(schizo_corr_test)), np.ones(len(healthy_corr_test))))









###############################################################################
###############################################################################
###############################################################################

####################################################################################
####################################################################################
####################################################################################
############################# Forward and Backward Feature Selection  ##############

# forward=True,floating=False   :  This is for "Forward Selection"             (
# forward=False,floating=False  :  This is for "Backward Selection"            
# forward=True,floating=True    :  This is for "Stepwise forward Regression" 
# forward=False,floating=True   :  This is for "Stepwise backward Regression"

####################################################################################

#importing the necessary libraries
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import SVC
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs



sfs = SFS(SVC(kernel ='rbf',C=1,gamma='auto'),
           k_features=100,                # no. of features to select
           forward=True,floating=True,      # This is for "Stepwise Selection",  For "Backward Selection" Make both of them False
           scoring = 'accuracy',            # for classification, it can be accuracy, precision, recall, f1-score, etc.
           cv =4, n_jobs=-1)                          # cv is k fold cross validation

 

sfs.fit(X, Y)
 
#sfs.k_score_
 
sfs_dataframe=pd.DataFrame.from_dict(sfs.get_metric_dict()).T
#sfs_dataframe.to_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\feature_ind_and_acc_for_cv_4_schizo_corr_t_test_159Regions_Stepwise_forward_selection_mlxtend.csv')



fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')

plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()
 
     
#    s=sfs.subsets_
#indexes=list(sfs.k_feature_idx_)

    

sfs_dataframe_sorted= sfs_dataframe.sort_values(by=['avg_score','feature_idx'], ascending=[False,False])
#acs_sorted= acs_sorted.reset_index()

sfs_dataframe_sorted.to_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\feature_ind_and_acc_for_cv_4_schizo_corr_t_test_159Regions_Stepwise_forward_selection_mlxtend.csv')

#sfs_frame_sorted=pd.read_csv(r'D:\Nitin Python\fMRI ROI\Schizo and Healthy\ROI files needed\feature_ind_and_acc_for_cv_4_schizo_corr_t_test_159Regions_Stepwise_forward_selection_mlxtend.csv')


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
#####################  Testing ##########################################

all_acc=[]
features_index=[]

X_train=X
Y_train=Y

Index_lists= list(sfs_dataframe_sorted.iloc[0:63,1].values)

for feat_index in Index_lists:


    ############################################
    X_train= X[:, list(map(int, feat_index[1:-1].split(', ')))]
    X_test1= X_test[:, list(map(int, feat_index[1:-1].split(', ')))]
    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test1= sc_X.transform(X_test1)
    
    
    from sklearn.svm import SVC
    svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
    svc_reg.fit(X_train,Y_train)

    Y_pred = svc_reg.predict(X_test1)

    
    from sklearn.metrics import confusion_matrix
    cmf= confusion_matrix(Y_test,Y_pred)
    
    from sklearn.metrics import accuracy_score
    acs=accuracy_score(Y_test,Y_pred)
 
    
    features_index.append(feat_index)
    
    all_acc.append((acs*100))
    





















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



















    
    
    















