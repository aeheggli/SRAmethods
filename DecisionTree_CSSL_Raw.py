# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:22:14 2021

@author: anne
"""
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import graphviz

pio.renderers.default='browser'

#%%IMPORT DATA FOR ML

#Raw data
os.chdir('/Users/anne/OneDrive/Data/QC_for_SRA/raw')
cssl_2006 = pd.read_csv("sh_428_WY2006.csv", parse_dates=['Date'])
cssl_2007 = pd.read_csv("sh_428_WY2007.csv", parse_dates=['Date'])
cssl_2008 = pd.read_csv("sh_428_WY2008.csv", parse_dates=['Date'])
cssl_2009 = pd.read_csv("sh_428_WY2009.csv", parse_dates=['Date'])
cssl_2010 = pd.read_csv("sh_428_WY2010.csv", parse_dates=['Date'])
cssl_2013 = pd.read_csv("sh_428_WY2013.csv", parse_dates=['Date'])
cssl_2014 = pd.read_csv("sh_428_WY2014.csv", parse_dates=['Date'])
cssl_2015 = pd.read_csv("sh_428_WY2015.csv", parse_dates=['Date'])
cssl_2016 = pd.read_csv("sh_428_WY2016.csv", parse_dates=['Date'])
cssl_2017 = pd.read_csv("sh_428_WY2017.csv", parse_dates=['Date'])
cssl_2018 = pd.read_csv("sh_428_WY2018.csv", parse_dates=['Date'])
cssl_2019 = pd.read_csv("sh_428_WY2019.csv", parse_dates=['Date'])

cssl = [cssl_2006, cssl_2007, cssl_2008, cssl_2009, cssl_2010, cssl_2013, cssl_2014, cssl_2015,cssl_2016, cssl_2017, cssl_2018, cssl_2019 ]

df = pd.concat(cssl)
df.columns = ['date','sh_temp_C','sh_temp_flag','sh_precip_mm','sh_precip_QC','sh_snowdepth_cm','sh_snowdepth_QC','sh_SWE_mm','sh_SWE_QC','sh_sm2_%','sh_sm2_QC','sh_sm8_%','sh_sm8_QC','sh_sm20_%','sh_sm20_QC','sh_st2_C','sh_st2_QC','sh_st8_C','sh_st2_QC','sh_st20_C','sh_st20_QC'] 

# df['precip_1h'] = df['sh_precip_manual_mm'].diff()
df['water_year'] = df.date.dt.year.where(df.date.dt.month < 10, df.date.dt.year + 1)
df['hour_of_wy'] = df.groupby([(df['water_year'] != df['water_year'].shift()).cumsum()]) \
                                  .cumcount() + 1
df = df.set_index('date')
df = df.asfreq('H')
df['month'] = df.index.month

#date	temp_C	temp_flag	precip_mm	precip_QC	snowdepth_cm	snowdepth_QC	SWE_mm	SWE_QC	SM2_%	SM2_QC	SM8_%	SM8_QC	SM20_%	SM20_QC	ST2_C	ST2_QC	ST8_C	ST2_QC	ST20_C	ST20_QC

#%% FEATURE ENGINEERING

##PRECIPITATION WINDOWS
df['precip_1h'] = df['sh_precip_mm'].diff()
df['precip_2h'] = df['precip_1h'].rolling(2).sum()
df['precip_3h'] = df['precip_1h'].rolling(3).sum()
df['precip_4h'] = df['precip_1h'].rolling(4).sum()
df['precip_5h'] = df['precip_1h'].rolling(5).sum()
df['precip_6h'] = df['precip_1h'].rolling(6).sum()
# df['precip_7h'] = df['precip_1h'].rolling(7).sum()
# df['precip_8h'] = df['precip_1h'].rolling(8).sum()
# df['precip_9h'] = df['precip_1h'].rolling(9).sum()
# df['precip_10h'] = df['precip_1h'].rolling(10).sum()
# df['precip_11h'] = df['precip_1h'].rolling(11).sum()
df['precip_12h'] = df['precip_1h'].rolling(12).sum() #turn on only for distribution plots
# df['precip_24h'] = df['precip_1h'].rolling(24).sum() #turn on only for distribution plots

##SNOW DEPTH WINDOWS
# df['delta_depth_1h'] = df['sh_snowdepth_cm'].diff()
df['depth_1h_earlier'] = df['sh_snowdepth_cm'].shift(1)
df['depth_2h_earlier'] = df['sh_snowdepth_cm'].shift(2)
df['depth_3h_earlier'] = df['sh_snowdepth_cm'].shift(3)
df['depth_4h_earlier'] = df['sh_snowdepth_cm'].shift(4)
df['depth_5h_earlier'] = df['sh_snowdepth_cm'].shift(5)
df['depth_6h_earlier'] = df['sh_snowdepth_cm'].shift(6)

# SWE CHANGE
df['delta_SWE_1h'] = df['sh_SWE_mm'].diff()
df['delta_SWE_2h'] = df['sh_SWE_mm'].diff(2)
df['delta_SWE_3h'] = df['sh_SWE_mm'].diff(3)
df['delta_SWE_4h'] = df['sh_SWE_mm'].diff(4)
df['delta_SWE_5h'] = df['sh_SWE_mm'].diff(5)
df['delta_SWE_6h'] = df['sh_SWE_mm'].diff(6)

##DENSITY WINDOWS
df['density_pct']=df['sh_SWE_mm']/df['sh_snowdepth_cm']*10
df['density_1h_earlier'] = df['density_pct'].shift(1)
df['density_2h_earlier'] = df['density_pct'].shift(2)
df['density_3h_earlier'] = df['density_pct'].shift(3)
df['density_4h_earlier'] = df['density_pct'].shift(4)
df['density_5h_earlier'] = df['density_pct'].shift(5)
df['density_6h_earlier'] = df['density_pct'].shift(6)

#DENSITY CHANGE
# df['delta_density_1h'] = df['density_pct'].diff()
# df['delta_density_2h'] = df['density_pct'].diff(2)
# df['delta_density_3h'] = df['density_pct'].diff(3)
# df['delta_density_4h'] = df['density_pct'].diff(4)
# df['delta_density_5h'] = df['density_pct'].diff(5)
# df['delta_density_6h'] = df['density_pct'].diff(6)

##TEMPERATURE WINDOWS
df['temp_1h_earlier'] = df['sh_temp_C'].shift(1)
df['temp_2h_earlier'] = df['sh_temp_C'].shift(2)
df['temp_3h_earlier'] = df['sh_temp_C'].shift(3)
df['temp_4h_earlier'] = df['sh_temp_C'].shift(4)
df['temp_5h_earlier'] = df['sh_temp_C'].shift(5)
df['temp_6h_earlier'] = df['sh_temp_C'].shift(6)

df['temp_2h_max'] = df['sh_temp_C'].rolling(2).max()
df['temp_3h_max'] = df['sh_temp_C'].rolling(3).max()
df['temp_4h_max'] = df['sh_temp_C'].rolling(4).max()
df['temp_5h_max'] = df['sh_temp_C'].rolling(5).max()
df['temp_6h_max'] = df['sh_temp_C'].rolling(6).max()
# df['temp_12h_max'] = df['sh_temp_C'].rolling(12).max() #turn on only for distribution plots

# # degree hours
# df['heating_hour'] = np.where(df['sh_temp_C'] > 0, 1, 0) 
# df['growingheating_hour'] = df.groupby([(df['heating_hour'] != df['heating_hour'].shift()).cumsum()]) \
#                                   .cumcount() + 1
# df.loc[df['heating_hour'] == 0, 'growingheating_hour'] = 0

# ##HOURS SINCE RAIN
# df['hours_since_rainfall'] = df.groupby([(df['rain'] != df['rain'].shift()).cumsum()]) \
#                                   .cumcount() + 1
# df.loc[df['rain'] == 1, 'hours_since_rainfall'] = 0

# ##HOURS OF CONSECUTUVE RAINFALL
# df['hours_of_rainfall'] = df.groupby([(df['rain'] != df['rain'].shift()).cumsum()]) \
#                                   .cumcount() + 1
# df.loc[df['rain'] == 0, 'hours_of_rainfall'] = 0

##SOIL MOISTURE
df['sh_sm2_auto_%'] = df['sh_sm2_%'].rolling(6, center=True, closed='right').median()
df['sh_sm8_auto_%'] = df['sh_sm8_%'].rolling(6, center=True, closed='right').median()
df['sh_sm20_auto_%'] = df['sh_sm20_%'].rolling(6, center=True, closed='right').median()

##SOIL MOISTURE CHANGE
df['delta_autosm2'] = df['sh_sm2_auto_%'].diff()
df['delta_autosm2_2h'] = df['sh_sm2_auto_%'].diff(2)
df['delta_autosm8'] = df['sh_sm8_auto_%'].diff()
df['delta_autosm8_2h'] = df['sh_sm8_auto_%'].diff(2)
df['delta_autosm20'] = df['sh_sm20_auto_%'].diff()
df['delta_autosm20_2h'] = df['sh_sm20_auto_%'].diff(2)

# df['delta_sm2'] = df['sh_sm2_%'].diff()
# df['delta_sm2_2h'] = df['sh_sm2_%'].diff(2)
# df['delta_sm8'] = df['sh_sm8_%'].diff()
# df['delta_sm8_2h'] = df['sh_sm8_%'].diff(2)
# df['delta_sm20'] = df['sh_sm20_%'].diff()
# df['delta_sm20_2h'] = df['sh_sm20_%'].diff(2)

##CREATE IDENTIFY WATER RELEASE
df['water_release1'] = np.where((df['delta_autosm2']>0.5) | (df['delta_autosm8']>0.5) | (df['delta_autosm20']>0.5), 1, 0)
df['water_release2'] = np.where((df['delta_autosm2_2h']>=1) | (df['delta_autosm8_2h']>=1) | (df['delta_autosm20_2h']>=1), 1, 0)
df['saturation'] = np.where((df['sh_sm2_auto_%']>=39) & (df['sh_sm8_auto_%']>=39) & (df['sh_sm20_auto_%']>=39), 1, 0)
df['water_release_target'] = np.where((df['water_release1'] == 1) | (df['water_release2'] == 1) | (df['saturation'] == 1), 1,0)



#%% FILTER DATA & CREATE TARGET
# df = df.set_index('Date')
df_ML = pd.concat([df.loc["2005-10-01 00:00:00":"2006-04-18 00:00:00"],
                    df.loc["2006-10-01 00:00:00":"2007-03-01 00:00:00"], #Defined by peak SWE
                    df.loc["2007-10-01 00:00:00":"2008-02-25 00:00:00"], #Defined by peak SWE
                    df.loc["2009-02-22 00:00:00":"2009-03-23 00:00:00"], #Defined by peak SWE and 8" sensor issues - following up with Jeff Anderson
                    df.loc["2009-10-01 00:00:00":"2010-04-14 00:00:00"], #Defined by peak SWE
                    df.loc["2012-10-01 00:00:00":"2013-03-10 00:00:00"], #Defined by peak SWE
                    df.loc["2013-10-01 00:00:00":"2014-03-11 00:00:00"], #Defined by 1st peak SWE
                    df.loc["2014-03-24 00:00:00":"2014-04-05 00:00:00"], #Defined by 2nd peak SWE
                    df.loc["2014-10-01 00:00:00":"2015-01-06 00:00:00"], #Defined by the 1st peak SWE
                    df.loc["2015-02-04 00:00:00":"2015-02-12 00:00:00"], #Defined by the 2nd peak SWE
                    df.loc["2015-10-01 00:00:00":"2016-03-16 00:00:00"], #Defined by peak SWE
                    df.loc["2016-10-01 00:00:00":"2017-04-18 00:00:00"], #Defined by peak SWE
                    df.loc["2017-10-01 00:00:00":"2018-03-27 00:00:00"], #Defined by peak SWE
                    df.loc["2018-10-01 00:00:00":"2019-04-03 00:00:00"] #Defined by peak SWE
                   ]) 
    
# DROP SUSPECT DATA - NA FOR RAW DATA

# FILTER FOR SNOW COVER 
df_ML = df_ML.loc[(df['sh_SWE_mm']>=100)]
df_ML = df_ML.loc[(df['water_release_target']==1)]

# ADD TARGET VARIABLES: ROS & WARM DAY MELT MANUALLY IDENTIFIED
os.chdir('/Users/anne/OneDrive/Data/QC_for_SRA/MLdata')
target = pd.read_csv("df_ML_ROSidentified.csv", parse_dates=['date'], index_col=['date'])
df_ML = pd.concat([df_ML, target], axis=1)

df_ML['unidentified'] = np.where((df_ML['warmday']==0) & (df_ML['ROS']==0), 1, 0)
df_ML = df_ML[df_ML.unidentified == 0]
print(df_ML.groupby('classification').count()) 
df_ML = df_ML.drop(['unidentified'], axis=1)
# print(df_ML.groupby('classification').count()) 

df_ML = df_ML.dropna(subset=['ROS'])
  
#%% DROP COLUMNS FOR ML
### STEP 1: CHEAT AND OVERFIT THE MODEL TO BE SURE THE CODE IS WORKING
###STEP 2: REMOVE FEATURES USED IN THE DEVELOPMENT OF THE TARGET

print(df_ML.columns)

### MUST DROP
df_ML = df_ML.drop(['sh_temp_flag', 'sh_precip_QC', 'sh_snowdepth_QC', 'sh_SWE_QC',
       'sh_sm2_%', 'sh_sm2_QC', 'sh_sm8_%', 'sh_sm8_QC', 'sh_sm20_%',
       'sh_sm20_QC', 'sh_st2_C', 'sh_st2_QC', 'sh_st8_C', 'sh_st2_QC',
       'sh_st20_C', 'sh_st20_QC', 'water_year', 'hour_of_wy', 'month',
       'sh_sm2_auto_%', 'sh_sm8_auto_%', 'sh_sm20_auto_%', 'delta_autosm2',
       'delta_autosm2_2h', 'delta_autosm8', 'delta_autosm8_2h',
       'delta_autosm20', 'delta_autosm20_2h', 'water_release1',
       'water_release2', 'saturation', 'water_release_target'], axis=1)

## PICK A TARGET
df_ML = df_ML.drop(['warmday'], axis=1)
# df_ML = df_ML.drop(['ROS'], axis=1)
df_ML = df_ML.drop(['classification'], axis=1)

##### MAY DROP
df_ML = df_ML.drop(['sh_precip_mm'], axis=1)

# df_ML = df_ML.drop(['precip_1h'], axis=1) #ROS vs WarmDay
# df_ML = df_ML.drop(['precip_2h'], axis=1) #ROS vs WarmDay
# df_ML = df_ML.drop(['precip_3h'], axis=1) #ROS vs WarmDay
# df_ML = df_ML.drop(['precip_4h'], axis=1) #ROS vs WarmDay
# df_ML = df_ML.drop(['precip_5h'], axis=1) #ROS vs WarmDay
# df_ML = df_ML.drop(['precip_6h'], axis=1) #ROS vs WarmDay
# df_ML = df_ML.drop(['precip_12h'], axis=1) #ROS vs WarmDay

df_ML = df_ML.drop(['sh_snowdepth_cm'], axis=1)
df_ML = df_ML.drop(['depth_1h_earlier'], axis=1)
df_ML = df_ML.drop(['depth_2h_earlier'], axis=1)
df_ML = df_ML.drop(['depth_3h_earlier'], axis=1)
df_ML = df_ML.drop(['depth_4h_earlier'], axis=1)
df_ML = df_ML.drop(['depth_5h_earlier'], axis=1)
df_ML = df_ML.drop(['depth_6h_earlier'], axis=1)

df_ML = df_ML.drop(['sh_SWE_mm'], axis=1)
df_ML = df_ML.drop(['delta_SWE_1h'], axis=1)
df_ML = df_ML.drop(['delta_SWE_2h'], axis=1)
df_ML = df_ML.drop(['delta_SWE_3h'], axis=1)
df_ML = df_ML.drop(['delta_SWE_4h'], axis=1)
df_ML = df_ML.drop(['delta_SWE_5h'], axis=1)
df_ML = df_ML.drop(['delta_SWE_6h'], axis=1)

df_ML = df_ML.drop(['density_pct'], axis=1)
df_ML = df_ML.drop(['density_1h_earlier'], axis=1)
df_ML = df_ML.drop(['density_2h_earlier'], axis=1)
df_ML = df_ML.drop(['density_3h_earlier'], axis=1)
df_ML = df_ML.drop(['density_4h_earlier'], axis=1)
df_ML = df_ML.drop(['density_5h_earlier'], axis=1)
df_ML = df_ML.drop(['density_6h_earlier'], axis=1)

df_ML = df_ML.drop(['sh_temp_C'], axis=1)
df_ML = df_ML.drop(['temp_1h_earlier'], axis=1)
df_ML = df_ML.drop(['temp_2h_earlier'], axis=1)
df_ML = df_ML.drop(['temp_3h_earlier'], axis=1)
df_ML = df_ML.drop(['temp_4h_earlier'], axis=1)
df_ML = df_ML.drop(['temp_5h_earlier'], axis=1)
df_ML = df_ML.drop(['temp_6h_earlier'], axis=1)

df_ML = df_ML.drop(['temp_2h_max'], axis=1)
df_ML = df_ML.drop(['temp_3h_max'], axis=1)
df_ML = df_ML.drop(['temp_4h_max'], axis=1)
df_ML = df_ML.drop(['temp_5h_max'], axis=1)
# df_ML = df_ML.drop(['temp_6h_max'], axis=1) #ROS vs WarmDay

df_ML = df_ML.fillna(-9999)

#%% SPLIT TEST AND TRAIN DATA

# # ###LEARN FROM 2006-2018 and TEST ON 2019
# X_train = df_ML["2005-10-01 00:00:00":"2018-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for training years
# y_train = df_ML.loc["2005-10-01 00:00:00":"2018-09-30 00:00:00", 'ROS'] #y = target for training years
# X_test = df_ML["2018-10-01 00:00:00":"2019-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
# y_test = df_ML.loc["2018-10-01 00:00:00":"2019-09-30 00:00:00", 'ROS'] #y = target for test years

###LEARN FROM 2007-2019 and TEST ON 2006-2007
X_train = df_ML["2007-10-01 00:00:00":"2019-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for training years
y_train = df_ML.loc["2007-10-01 00:00:00":"2019-09-30 00:00:00", 'ROS'] #y = target for training years
X_test = df_ML["2005-10-01 00:00:00":"2007-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
y_test = df_ML.loc["2005-10-01 00:00:00":"2007-09-30 00:00:00", 'ROS'] #y = target for test years

# ##Create feature columns - print to list, copy/paste and remove the target variable
feature_cols = X_train.columns.tolist()
# print(feature_cols)

#%%    
maxdepth = 4

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=maxdepth, criterion='gini', splitter='best', max_features=None)
# min_samples_leaf=30, min_samples_split=2,

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train) 

# Model Scores
a = metrics.accuracy_score(y_test, y_pred)
p = metrics.precision_score(y_test, y_pred)
r = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
f2 = metrics.fbeta_score(y_test, y_pred, beta=2.0)
print('Result on test: a=%.3f, p=%.3f, r=%.3f, f1=%.3f, f2=%.3f' % (a, p, r, f1, f2))
at = metrics.accuracy_score(y_train, y_pred_train)
pt = metrics.precision_score(y_train, y_pred_train)
rt = metrics.recall_score(y_train, y_pred_train)
f1t = metrics.f1_score(y_train, y_pred_train)
f2t = metrics.fbeta_score(y_train, y_pred_train, beta=2.0)
print('Result on train: at=%.3f, pt=%.3f, rt=%.3f, f1t=%.3f, f2t=%.3f' % (at, pt, rt, f1t, f2t))

# # CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
# cm_train = confusion_matrix(y_train, y_pred_train)
# print(cm_train)
# cm_test = confusion_matrix(y_test, y_pred)
# print(cm_test)

# fig, axes = plt.subplots(1,2,figsize=(12,6))
# sns.heatmap(confusion_matrix(y_train, y_pred_train), fmt=".0f", annot=True, ax=axes[0])
# axes[0].set(xlabel="Actual",ylabel="Predicted")
# axes[0].title.set_text('c. Raw Training Data')
# sns.heatmap(confusion_matrix(y_test, y_pred), fmt=".0f", annot=True,  ax=axes[1])
# axes[1].set(xlabel="Actual",ylabel="Predicted")
# axes[1].title.set_text('d. Raw Test Data')
# plt.suptitle(maxdepth)
# plt.show()

fig = plt.figure(figsize=(4, 3), dpi = 300)
sns.heatmap(confusion_matrix(y_train, y_pred_train), fmt=".0f", annot=True)
plt.xlabel("Actual", size=12)
plt.ylabel("Predicted", size=12)
# plt.title("b. Raw Data", size=14)

#%% Decision trees
# fig = plt.figure(dpi = 300)
dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=feature_cols,
                      max_depth=maxdepth,
                      class_names=['warm_day', 'ROS'],  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
# graph.render("Water Release 4")
graph.format = 'png'
graph.render('dtree_render',view=True)


from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
import graphviz
from dtreeviz.trees import dtreeviz # remember to load the package

class_colors = [None, # 0 classes
                None, # 1 class
                ["#FFB732","#5E5EFF"], # 2 classes
                ["#ff0000","#00ff00",'#0000ff'], # 3 classes
]

viz = dtreeviz(clf, X_train, y_train,
                target_name="ROS",
                feature_names=feature_cols,
                class_names=['warm_day', 'ROS'], colors = {'classes': class_colors})
viz.view() 


#%% get importance
importance = clf.feature_importances_
indices = np.argsort(importance)

fig, axes = plt.subplots(1, 1, figsize=(12, 14))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

#%%
tree_clas = DecisionTreeClassifier(criterion='gini', splitter='best', max_features=None)
# criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, ccp_alpha=0.0

param_grid = {
                'max_depth' : np.arange(3,7),
                # 'criterion' :['gini'],
                'min_samples_split' : np.arange(2,40),
                'min_samples_leaf' : np.arange(2,40)
             }

print("start fitting the data")
create_grid = GridSearchCV(tree_clas, param_grid=param_grid, cv=5)
create_grid.fit(X_train, y_train)
print(create_grid.score(X_train, y_train))
print("!!! best fit parameters from GridSearchCV!!!!")
print(create_grid.best_params_)
