schizo_series_all222=pd.DataFrame( schizo_series_all16)
healthy_series_all222=pd.DataFrame(healthy_series_all16)


schizo_roi_all222 =pd.DataFrame(schizo_series)
healthy_roi_all222= pd.DataFrame(healthy_series)


##############################################################

result1 = pd.concat([schizo_series_all222, schizo_roi_all222], axis=1, sort=False)
result2 = pd.concat([healthy_series_all222, healthy_roi_all222], axis=1, sort=False)

#############################################################################


#df = result1.append(result2)
df = schizo_series_all222.append(healthy_series_all222)

df= AB_inte[0]

df= df.reset_index(drop=True)



total_features=len(df.columns)



feat_order=[65,  1,  3,  0, 66,  2, 31, 20, 25,  5,  9, 16, 12, 59, 19,  6, 26,
       48, 22, 56, 52, 28, 30, 51, 50, 53, 61, 35, 64,  7, 60, 34, 36, 27,
       17, 10, 55, 15,  8, 46,  4, 63, 18, 62, 32, 13, 44, 29, 40, 21, 42,
       38, 23, 49, 39, 58, 54, 45, 33, 57, 47, 24, 41, 37, 11, 43, 14]
    

all_acc=[]

for i in reversed(range(1,341)):

###########################################################################
############ Corr t-Test statrt it from here
df=main_corr_frame
    
X= df.iloc[:,:].values
#[3,0,19,11,22,26,23,2,8,9,20,6,13,16,21,27,7,12,17,4,14,24,25,18,15,5,1]
################################################
######### Nor 23.6 sec
#X= np.array(AB_inte[0])
#X= X.reshape(-1, 1)

#Y = np.concatenate((np.zeros(len(schizo_series_all222)), np.ones(len(healthy_series_all222))))

Y = np.concatenate((np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))))
#Y = np.concatenate((np.zeros(shape=855), np.ones(shape=855)))

############################################################

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)



acs=[]

for i in range(1000):
    from sklearn.model_selection import train_test_split
    X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.2)
    
    from sklearn.svm import SVC
    svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
    svc_reg.fit(X_train,Y_train)

    Y_pred = svc_reg.predict(X_test)

    
#    from sklearn.metrics import confusion_matrix
#    cmf= confusion_matrix(Y_test,Y_pred)
    
    from sklearn.metrics import accuracy_score
    acs.append(accuracy_score(Y_test,Y_pred))


acs1 =np.mean(acs) 
acs1=acs1*100

std=np.std(acs)
std=std*100

print(acs1,std)



#####################################################################################

##### FOR LINEAR KERNAL ONLY

coef=pd.DataFrame(abs(svc_reg.coef_[0]),columns=['coef'])

sort_coef= coef.sort_values('coef',ascending=False)
feature_imp=sort_coef.index
np.transpose(feature_imp)



'''
features_names=[]
for i in np.linspace(0,total_features-1,total_features,dtype=np.int32):
    features_names.append(str(i))


def f_importances(coef, names):
    imp = coef
    zipped=zip(names,imp)
    imp,names = sorted(zipped, key = lambda t: t[1])
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


f_importances(svm.coef_, features_names)


names=features_names

coef=abs(svc_reg.coef_[0])

'''

