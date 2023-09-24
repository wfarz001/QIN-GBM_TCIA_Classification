# -*- coding: utf-8 -*-
"""

@author: Walia farzana
"""
#########Importing the Libaries#################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV
import matplotlib.pyplot as plt
#import seaborn as sns
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1234)
from scipy import stats

######Data Loading###################

data_df=pd.read_csv('Combined_REP.csv')
data_df = data_df.sample(frac = 1) ## Data Shuffling


texture=pd.read_csv('Texture_REP.csv')
area_vol=pd.read_csv('Area_Volume_REP.csv')
histogram=pd.read_csv('Histogram_REP.csv')

data_df=data_df.dropna()

X_train=data_df.iloc[:,2:1044]
 
y_train=data_df.iloc[:,:2]
y_train=y_train.iloc[:,-1:]


feature=pd.concat([texture,area_vol],axis=0).T
feature=feature.values.tolist()

intersection_cols = X_train.columns & feature[-1]

X_train=X_train[intersection_cols]


########Variance Threshold feature filtering########################
cor=X_train.corr().abs()
upper=cor.where(np.triu(np.ones(cor.shape),k=1).astype(np.bool))
to_drop=[column for column in upper.columns if any(upper [column]>0.95)]

X_train.drop(to_drop,axis=1,inplace=True)


col=list(X_train)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_train=pd.DataFrame(X_train,columns=col)

count=0
B=10 # number of iterations
auc_list=[]
precision_list=[]
f1_list=[]
recall_list=[]
acc_list=[]

y_true_list=[]
pred_list=[]

selected_feature_list=[]
for n in range(0,B):
    
    from sklearn.feature_selection import SelectKBest ###K-Best Feature Filtering 

    from sklearn.feature_selection import f_classif

    bestfeatures = SelectKBest(score_func=f_classif, k=150)
    fit = bestfeatures.fit(X_train,y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)

    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']

    m=featureScores.nlargest(150,'Score')
    selected_feature=m.iloc[:,:-1]
    selected_feature=selected_feature.set_index('Specs')
    slected_feature=selected_feature.T
    intersection_cols = X_train.columns & slected_feature.columns
    X_train =X_train[intersection_cols]

    columnNames3=list(X_train)
    
    from sklearn.ensemble import RandomForestClassifier ###Recursive Feature Elimination

    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFE
    from sklearn.feature_selection import RFECV 

          ###Recursive Feature Elimination
     
    estimator=RandomForestClassifier(random_state=1234)
    sel=RFECV(estimator,step=1, cv=StratifiedKFold(5),scoring='accuracy')
    sel.fit(X_train,y_train)
    dset = pd.DataFrame()
    k = sel.get_support(1) #the most important features
    sel_features = X_train[X_train.columns[k]]
    columnNames2=list(sel_features)
    dset['attr'] = columnNames2
    dset['importance'] = sel.estimator_.feature_importances_
    dset = dset.sort_values(by='importance', ascending=False)
    dset_selected=dset.head(15)
    dset_selected_1=dset_selected.iloc[:,:-1]
    dset_selected_1=pd.DataFrame(dset_selected_1['attr'])
    dset_selected_1=dset_selected_1.set_index('attr')
    dset_selected_1=dset_selected_1.T
    intersection_cols = X_train.columns & dset_selected_1.columns
    X_rfe = X_train[intersection_cols]

    selected_feature=list(X_rfe)
    
    selected_feature_list.append(selected_feature)
    
    X_rfe=np.array(X_rfe)
    
    #######One-Class SVM model#########################
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score,recall_score, precision_score

    #from algorithm import SelfAdaptiveShifting
    from random import sample
    y_value = np.unique(y_train)

    f_index = np.where(y_train == y_value[0])[0]
    s_index = np.where(y_train == y_value[1])[0]


    target_X, target_y = X_rfe[f_index], np.ones(len(f_index)) #class level 0 as 1 to denote inlier
    outlier_X, outlier_y = X_rfe[s_index], -np.ones(len(s_index)) #class level 1 as -1 to denote outlier

    # target_X_train, target_X_test, target_y_train, target_y_test = train_test_split(target_X, target_y, shuffle=True,
    #                                                                                     random_state=42+n, test_size=0.2)
    target_X=pd.DataFrame(target_X)
    target_y=pd.DataFrame(target_y)
    target_data=pd.concat([target_X,target_y],axis=1)
    
    target_data=np.array(target_data)
    
    dim=target_data.shape[0]
    l=[i for i in range(dim)]
    flag=sample(l,34)
    #print(flag)
    flag1=list(set(l)-set(flag))
    #print(flag1)  
    
    target_train=target_data[flag]
    target_test=target_data[flag1]
    
    target_train=pd.DataFrame(target_train)
    target_test=pd.DataFrame(target_test)
    
    target_X_train=np.array(target_train.iloc[:,:-1])
    target_y_train=np.array(target_train.iloc[:,-1:])
    
    target_X_test=np.array(target_test.iloc[:,:-1])
    target_y_test=np.array(target_test.iloc[:,-1:])
    
    count=count+1
    print('The number of the iteration:',count)

    # self_adaptive_shifting = SelfAdaptiveShifting(target_X_train)
    # self_adaptive_shifting.edge_pattern_detection(0.01)
    # pseudo_outlier_X = self_adaptive_shifting.generate_pseudo_outliers()
    # pseudo_target_X = self_adaptive_shifting.generate_pseudo_targets()
    # pseudo_outlier_y = -np.ones(len(pseudo_outlier_X))
    # pseudo_target_y = np.ones(len(pseudo_target_X))

    # gamma_candidates = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3, 1/np.size(target_X, -1)]
    # nu_candidates = [0.005, 0.01, 0.05, 0.1, 0.5]

    best_err = 1.0
    best_gamma, best_nu = 1/np.size(target_X, -1), 0.5

    # for gamma in tqdm(gamma_candidates):
    #     for nu in tqdm(nu_candidates):
    #         model = OneClassSVM(gamma=gamma, nu=nu).fit(target_X_train)
    #         err_o = 1 - np.mean(model.predict(pseudo_outlier_X) == pseudo_outlier_y)
    #         err_t = 1 - np.mean(model.predict(pseudo_target_X) == pseudo_target_y)
    #         err = float((err_o + err_t) / 2)
    #         if err < best_err:
    #                 best_err = err
    #                 best_gamma = gamma
    #                 best_nu = nu
                    
    best_model = OneClassSVM(kernel='rbf', gamma=best_gamma, nu=best_nu).fit(target_X_train)
   

    from sklearn.metrics import roc_curve, auc

    
    k=np.r_[target_X_test,outlier_X]
    l=np.array([1]*9+[-1]*4)

    df_1=pd.DataFrame(k)
    df_2=pd.DataFrame(l)

    final_df=pd.concat([df_1,df_2],axis=1)

    df_s=final_df.sample(frac = 1) ## Data Shuffling

    X_test=df_s.iloc[:,:-1]
    y_true=df_s.iloc[:,-1:]

    X_test=np.array(X_test)
    y_true=np.array(y_true)
    
    y_true_list.append(y_true)

    pred=best_model.predict(X_test)
    
    pred_list.append(pred)

    fpr,tpr,thresholds =roc_curve(y_true,pred)

    roc_auc=auc(fpr,tpr)
    auc_list.append(roc_auc)
    
    #print('AUC:',roc_auc)

    precision=precision_score(y_true,pred, pos_label=-1,average="binary")
    precision_list.append(precision)
    
    #print('Precision:',precision)

    f1 = f1_score(y_true,pred, average="binary")
    f1_list.append(f1)
    
    #print('f1_score:',f1)

    recall=recall_score(y_true,pred, average="binary")
    recall_list.append(recall)
    
    #print('Recall:',recall)

    acc = accuracy_score(y_true,pred)
    acc_list.append(acc)
    
    #print('accuracy:',acc)
    
    #############Printing the Results#############################
    
def Average(lst):
     return sum(lst) / len(lst)   

final_AUC=Average(auc_list)
final_precision=Average(precision_list)
final_recall=Average(recall_list)
final_f1_score=Average(f1_list)
final_accuracy=Average(acc_list)

print('The average result after total iteration Number:',B)
#print('Final AUC:',final_AUC)
print('Final_TPR:',final_precision)
print('Final FPR:',final_recall)
#print('Final F1-score:',final_f1_score)
print('Final Accuracy:',final_accuracy)


