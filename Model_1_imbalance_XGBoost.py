# -*- coding: utf-8 -*-
"""

@author: Walia_farzana
"""
###########Importing Necessary Libaries########################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV
import matplotlib.pyplot as plt
#import seaborn as sns
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
from scipy import stats


######Data Loading#########
data_df=pd.read_csv('Combined_REP.csv')
data_df = data_df.sample(frac = 1) ## Data Shuffling


texture=pd.read_csv('Texture_REP.csv')
area_vol=pd.read_csv('Area_Volume_REP.csv')
histogram=pd.read_csv('Histogram_REP.csv')


data_df=data_df.dropna()

X_train=data_df.iloc[:,2:1044]
 
y_train=data_df.iloc[:,:2]
y_train=y_train.iloc[:,-1:]
y_train['REP_Status'] = y_train['REP_Status'].map({1:'Progression', 0:'No Progression'})


from sklearn.preprocessing import LabelEncoder
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)

y_train = label_encoder.transform(y_train)



feature=pd.concat([texture,area_vol],axis=0).T
feature=feature.values.tolist()

intersection_cols = X_train.columns & feature[-1]

X_train=X_train[intersection_cols]

####Feature Selection Process############################3

cor=X_train.corr().abs()
upper=cor.where(np.triu(np.ones(cor.shape),k=1).astype(np.bool))
to_drop=[column for column in upper.columns if any(upper [column]>0.95)]

X_train.drop(to_drop,axis=1,inplace=True)


col=list(X_train)

######Data Scaling########
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler() 
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_train=pd.DataFrame(X_train,columns=col)

###Over-Sampling & Under-Sampling##############

from collections import Counter
# summarize class distribution
counter = Counter(y_train)
print('Before sampling the class distribution:',counter)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# define pipeline
over = SMOTE(sampling_strategy=0.3,k_neighbors=3)
under = RandomUnderSampler(sampling_strategy=0.4)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
X, y = pipeline.fit_resample(X_train, y_train)



counter = Counter(y)
print('Afrer sampling class distribution:',counter)

y=pd.DataFrame(y)

data=pd.concat([X,y],axis=1)
data = data.sample(frac = 1) ## Data Shuffling

y_res=pd.DataFrame()
X=data.iloc[:,:-1]
y_res['REP_Status']=data.iloc[:,-1:]

y_res['REP_Status'] = y_res['REP_Status'].map({1:'Progression', 0:'No Progression'})


from sklearn.preprocessing import LabelEncoder
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_res)

y_res = label_encoder.transform(y_res)



######K-Best Feature Selection###################
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

bestfeatures = SelectKBest(score_func=f_classif, k=150)
fit = bestfeatures.fit(X,y_res)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']

m=featureScores.nlargest(150,'Score')
selected_feature=m.iloc[:,:-1]
selected_feature=selected_feature.set_index('Specs')
slected_feature=selected_feature.T
intersection_cols = X.columns & slected_feature.columns
X =X[intersection_cols]

columnNames3=list(X)



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV 



estimator=RandomForestClassifier(random_state=101)
sel=RFECV(estimator,step=1, cv=StratifiedKFold(5),scoring='accuracy')
sel.fit(X,y_res)
dset = pd.DataFrame()
k = sel.get_support(1) #the most important features
sel_features = X[X.columns[k]]
columnNames2=list(sel_features)
dset['attr'] = columnNames2
dset['importance'] = sel.estimator_.feature_importances_
dset = dset.sort_values(by='importance', ascending=False)
dset_selected=dset.head(15) #number of selected features
dset_selected_1=dset_selected.iloc[:,:-1]
dset_selected_1=pd.DataFrame(dset_selected_1['attr'])
dset_selected_1=dset_selected_1.set_index('attr')
dset_selected_1=dset_selected_1.T
intersection_cols = X.columns & dset_selected_1.columns
X_rfe = X[intersection_cols]

selected_feature=list(X_rfe)

############### Imbalance XGBoost Model#######################################
#X_rfe=np.array(X_rfe)
import functools
from sklearn.metrics import make_scorer
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sklearn.model_selection import GridSearchCV
import statistics
xgboster_focal = imb_xgb(special_objective='focal')



X_rfe=np.array(X_rfe)
CV_focal_booster =  GridSearchCV(xgboster_focal, {"focal_gamma":[1.0,1.5,2.0,2.5,3.0]})
CV_focal_booster.fit(X_rfe, y_res)
xgboost_opt_param = CV_focal_booster.best_params_

xgboost_opt = imb_xgb(special_objective='focal', **xgboost_opt_param)


loo_splitter=StratifiedKFold(5)
mode=['accuracy','recall','precision']
accuracy_eval_func = functools.partial(xgboost_opt.score_eval_func, mode='accuracy')
#cross validation
loo_info_accuracy = cross_validate(xgboost_opt, X=X_rfe, y=y_res, cv=loo_splitter, scoring=make_scorer(accuracy_eval_func))

overall_accuracy=statistics.mean(loo_info_accuracy['test_score'])

TP_eval_func = functools.partial(xgboost_opt.correct_eval_func, mode='TP')
TN_eval_func = functools.partial(xgboost_opt.correct_eval_func, mode='TN')
FP_eval_func = functools.partial(xgboost_opt.correct_eval_func, mode='FP')
FN_eval_func = functools.partial(xgboost_opt.correct_eval_func, mode='FN')

loo_info_tn = cross_validate(xgboost_opt, X=X_rfe, y=y_res,cv=loo_splitter, scoring=make_scorer(TN_eval_func))
overall_tn =np.sum(loo_info_tn['test_score']).astype('float')

loo_info_tp = cross_validate(xgboost_opt, X=X_rfe, y=y_res,cv=loo_splitter, scoring=make_scorer(TP_eval_func))
overall_tp =np.sum(loo_info_tp['test_score']).astype('float')

loo_info_fp = cross_validate(xgboost_opt, X=X_rfe, y=y_res,cv=loo_splitter, scoring=make_scorer(FP_eval_func))
overall_fp =np.sum(loo_info_fp['test_score']).astype('float')

loo_info_fn = cross_validate(xgboost_opt, X=X_rfe, y=y_res,cv=loo_splitter, scoring=make_scorer(FN_eval_func))
overall_fn =np.sum(loo_info_fn['test_score']).astype('float')

tpr=overall_tp/(overall_tp+overall_fn)

fpr=overall_tn/(overall_tn+overall_fp)

############Print the Results###################################3

print('The number of features',len(selected_feature))
print('Accuracy:',overall_accuracy)
print('True Postive Rate:',tpr)
print('False Positive Rate:',fpr)
