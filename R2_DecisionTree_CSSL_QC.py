# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:22:14 2021

@author: anne
"""
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import graphviz

############################################################
######### SET PLOTTING DEFAULTS ###########################
# palette = sns.color_palette("colorblind")
# sns.palplot(palette)
palette = "colorblind"
plt.rcParams['lines.linewidth'] = 2

SMALL_SIZE = 8
MEDIUM_SIZE = 12
LARGE_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

#%%IMPORT DATA FOR ML

### Final QC Data ###
os.chdir('/Users/anne/OneDrive/Data/QC_for_SRA/MLdata')
cssl_2006 = pd.read_csv("428_WY2006_ML.csv", parse_dates=['date']).iloc[:, 1:]
cssl_2007 = pd.read_csv("428_WY2007_ML.csv", parse_dates=['date']).iloc[:, 1:]
cssl_2008 = pd.read_csv("428_WY2008_ML.csv", parse_dates=['date']).iloc[:, 1:]
cssl_2009 = pd.read_csv("428_WY2009_ML.csv", parse_dates=['date']).iloc[:, 1:]
cssl_2010 = pd.read_csv("428_WY2010_ML.csv", parse_dates=['date']).iloc[:, 1:]
cssl_2013 = pd.read_csv("428_WY2013_ML.csv", parse_dates=['date']).iloc[:, 1:]
cssl_2014 = pd.read_csv("428_WY2014_ML.csv", parse_dates=['date']).iloc[:, 1:]
cssl_2015 = pd.read_csv("428_WY2015_ML.csv", parse_dates=['date']).iloc[:, 1:]
cssl_2016 = pd.read_csv("428_WY2016_ML.csv", parse_dates=['date']).iloc[:, 1:]
cssl_2017 = pd.read_csv("428_WY2017_ML.csv", parse_dates=['date']).iloc[:, 1:]
cssl_2018 = pd.read_csv("428_WY2018_ML.csv", parse_dates=['date']).iloc[:, 1:]
cssl_2019 = pd.read_csv("428_WY2019_ML.csv", parse_dates=['date']).iloc[:, 1:]

cssl = [cssl_2006, cssl_2007, cssl_2008, cssl_2009, cssl_2010, cssl_2013, cssl_2014, cssl_2015, cssl_2016, cssl_2017, cssl_2018, cssl_2019 ] 
df = pd.concat(cssl)

# df = df.set_index('date')

#%% FEATURE ENGINEERING

##PRECIPITATION WINDOWS
df['precip_1h'] = df['sh_precip_manual_mm'].diff()
df['precip_2h'] = df['precip_1h'].rolling(2).sum()
df['precip_3h'] = df['precip_1h'].rolling(3).sum()
df['precip_4h'] = df['precip_1h'].rolling(4).sum()
df['precip_5h'] = df['precip_1h'].rolling(5).sum()
df['precip_6h'] = df['precip_1h'].rolling(6).sum()
df['precip_12h'] = df['precip_1h'].rolling(12).sum() #turn on only for distribution plots

##SNOW DEPTH WINDOWS
# df['delta_depth_1h'] = df['sh_depth_manual_cm'].diff()
df['depth_1h_earlier'] = df['sh_depth_manual_cm'].shift(1)
df['depth_2h_earlier'] = df['sh_depth_manual_cm'].shift(2)
df['depth_3h_earlier'] = df['sh_depth_manual_cm'].shift(3)
df['depth_4h_earlier'] = df['sh_depth_manual_cm'].shift(4)
df['depth_5h_earlier'] = df['sh_depth_manual_cm'].shift(5)
df['depth_6h_earlier'] = df['sh_depth_manual_cm'].shift(6)

# SWE CHANGE
df['delta_SWE_1h'] = df['sh_SWE_manual_mm'].diff()
df['delta_SWE_2h'] = df['sh_SWE_manual_mm'].diff(2)
df['delta_SWE_3h'] = df['sh_SWE_manual_mm'].diff(3)
df['delta_SWE_4h'] = df['sh_SWE_manual_mm'].diff(4)
df['delta_SWE_5h'] = df['sh_SWE_manual_mm'].diff(5)
df['delta_SWE_6h'] = df['sh_SWE_manual_mm'].diff(6)

##DENSITY WINDOWS
df['density_pct']=df['sh_SWE_manual_mm']/df['sh_depth_manual_cm']*10
df['density_1h_earlier'] = df['density_pct'].shift(1)
df['density_2h_earlier'] = df['density_pct'].shift(2)
df['density_3h_earlier'] = df['density_pct'].shift(3)
df['density_4h_earlier'] = df['density_pct'].shift(4)
df['density_5h_earlier'] = df['density_pct'].shift(5)
df['density_6h_earlier'] = df['density_pct'].shift(6)

##TEMPERATURE WINDOWS
df['temp_1h_earlier'] = df['sh_tempcorrected_C'].shift(1)
df['temp_2h_earlier'] = df['sh_tempcorrected_C'].shift(2)
df['temp_3h_earlier'] = df['sh_tempcorrected_C'].shift(3)
df['temp_4h_earlier'] = df['sh_tempcorrected_C'].shift(4)
df['temp_5h_earlier'] = df['sh_tempcorrected_C'].shift(5)
df['temp_6h_earlier'] = df['sh_tempcorrected_C'].shift(6)

df['temp_2h_max'] = df['sh_tempcorrected_C'].rolling(2).max()
df['temp_3h_max'] = df['sh_tempcorrected_C'].rolling(3).max()
df['temp_4h_max'] = df['sh_tempcorrected_C'].rolling(4).max()
df['temp_5h_max'] = df['sh_tempcorrected_C'].rolling(5).max()
df['temp_6h_max'] = df['sh_tempcorrected_C'].rolling(6).max()
# df['temp_12h_max'] = df['sh_tempcorrected_C'].rolling(12).max() #turn on only for distribution plots

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

##CREATE IDENTIFY WATER RELEASE
df['water_release1'] = np.where((df['delta_autosm2']>0.5) | (df['delta_autosm8']>0.5) | (df['delta_autosm20']>0.5), 1, 0)
df['water_release2'] = np.where((df['delta_autosm2_2h']>=1) | (df['delta_autosm8_2h']>=1) | (df['delta_autosm20_2h']>=1), 1, 0)
df['saturation'] = np.where((df['sh_sm2_auto_%']>=39) & (df['sh_sm8_auto_%']>=39) & (df['sh_sm20_auto_%']>=39), 1, 0)
df['water_release_target'] = np.where((df['water_release1'] == 1) | (df['water_release2'] == 1) | (df['saturation'] == 1), 1,0)


#%% FILTER DATA & CREATE TARGET
df = df.set_index('date')
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
    
# DROP SUSPECT DATA
df_ML = df_ML.loc[(df_ML['sh_SWE_manual_qcflag']!= "S")]
df_ML = df_ML.loc[(df_ML['sh_depth_manual_qcflag']!= "S")]
df_ML = df_ML.loc[(df_ML['sh_precip_manual_qcflag']!= "S")]

# FILTER FOR SNOW COVER 
df_ML = df_ML.loc[(df_ML['sh_SWE_manual_mm']>=100)]
df_ML = df_ML.loc[(df_ML['water_release_target']==1)]

# ADD TARGET VARIABLES: ROS & WARM DAY MELT MANUALLY IDENTIFIED
target = pd.read_csv("df_ML_ROSidentified.csv", parse_dates=['date'], index_col=['date'])
df_ML = pd.concat([df_ML, target], axis=1)
print(df_ML.groupby('classification').count())
      
df_ML['unidentified'] = np.where((df_ML['warmday']==0) & (df_ML['ROS']==0), 1, 0)
df_ML = df_ML[df_ML.unidentified == 0]
print(df_ML.groupby('unidentified').count()) 
print(df_ML.groupby('classification').count()) 
df_ML = df_ML.drop(['unidentified'], axis=1)
  
#%% DROP COLUMNS FOR ML
### STEP 1: CHEAT AND OVERFIT THE MODEL TO BE SURE THE CODE IS WORKING
###STEP 2: REMOVE FEATURES USED IN THE DEVELOPMENT OF THE TARGET

### MUST DROP
df_ML = df_ML.drop(['sh_SWE_manual_qcflag','sh_SWE_manual_qaflag','sh_depth_manual_qcflag','sh_depth_manual_qaflag','sh_precip_manual_qcflag',
                    'sh_precip_manual_qaflag','w10_precip_manual_qcflag', 'w10_precip_manual_qaflag','sh_temp_qcflag','sh_temp_qaflag',
                    'sh_sm2_%','sh_sm2_qc','sh_sm8_%','sh_sm8_qc','sh_sm20_%','sh_sm20_qc','sh_sm2_C_qaflag','sh_sm8_C_qaflag','sh_sm20_C_qaflag',
                    'sh_st2_C','sh_st2_qc','sh_st8_C','sh_st2_qc.1','sh_st20_C','sh_st20_qc','sh_st2_C_qaflag','sh_st8_C_qaflag','sh_st20_C_qaflag','sh_sm2_auto_%','sh_sm8_auto_%','sh_sm20_auto_%',
                    'delta_autosm2','delta_autosm2_2h','delta_autosm8','delta_autosm8_2h','delta_autosm20', 'delta_autosm20_2h', 'water_release1','water_release2','saturation'], axis=1)

## PICK A TARGET
df_ML = df_ML.drop(['warmday'], axis=1)
# df_ML = df_ML.drop(['ROS'], axis=1)
df_ML = df_ML.drop(['classification'], axis=1)

##### MAY DROP
df_ML = df_ML.drop(['sh_precip_manual_mm'], axis=1)
df_ML = df_ML.drop(['w10_precip_manual_mm'], axis=1)

df_ML = df_ML.drop(['sh_depth_manual_cm'], axis=1)
df_ML = df_ML.drop(['depth_1h_earlier'], axis=1)
df_ML = df_ML.drop(['depth_2h_earlier'], axis=1)
df_ML = df_ML.drop(['depth_3h_earlier'], axis=1)
df_ML = df_ML.drop(['depth_4h_earlier'], axis=1)
df_ML = df_ML.drop(['depth_5h_earlier'], axis=1)
df_ML = df_ML.drop(['depth_6h_earlier'], axis=1)

df_ML = df_ML.drop(['sh_SWE_manual_mm'], axis=1)
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

df_ML = df_ML.drop(['sh_tempcorrected_C'], axis=1)
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

df_ML = df_ML.drop(['water_release_target'], axis=1)

df_ML = df_ML.fillna(-9999)

#%% SPLIT TEST AND TRAIN DATA
############ UNOCMMENT EACH SPLIT FOR CROSS VALIDATION  ##################

####### TOWARDS SNOWPACK RUNOFF SELECTED MODEL ##################

###LEARN FROM 2008-2019 and TEST ON 2006-2007
X_train = df_ML["2007-10-01 00:00:00":"2019-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for training years
y_train = df_ML.loc["2007-10-01 00:00:00":"2019-09-30 00:00:00", 'ROS'] #y = target for training years
X_test = df_ML["2005-10-01 00:00:00":"2007-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
y_test = df_ML.loc["2005-10-01 00:00:00":"2007-09-30 00:00:00", 'ROS'] #y = target for test years

# ###LEARN FROM 2006 and 2009-2019 and 2016-2019 and TEST ON 2007 and 2008
# X_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2006-09-30 00:00:00"], 
#                       df_ML.loc["2008-10-01 00:00:00":"2019-09-30 00:00:00"]]).drop(['ROS'], axis = 1) #X = features for training years
# y_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2006-09-30 00:00:00"], 
#                       df_ML.loc["2008-10-01 00:00:00":"2019-09-30 00:00:00"]]) #y = target for training years
# y_train = pd.Series(y_train.ROS)
# X_test = df_ML["2006-10-01 00:00:00":"2008-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
# y_test = df_ML.loc["2006-10-01 00:00:00":"2008-09-30 00:00:00", 'ROS'] #y = target for test years

# ###LEARN FROM 2006-2007 and 2010-2019 and 2016-2019 and TEST ON 2008 and 2009
# X_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2007-09-30 00:00:00"], 
#                       df_ML.loc["2009-10-01 00:00:00":"2019-09-30 00:00:00"]]).drop(['ROS'], axis = 1) #X = features for training years
# y_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2007-09-30 00:00:00"], 
#                       df_ML.loc["2009-10-01 00:00:00":"2019-09-30 00:00:00"]]) #y = target for training years
# y_train = pd.Series(y_train.ROS)
# X_test = df_ML["2007-10-01 00:00:00":"2009-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
# y_test = df_ML.loc["2007-10-01 00:00:00":"2009-09-30 00:00:00", 'ROS'] #y = target for test years

# ###LEARN FROM 2006-2008 and 2013-2019 and 2016-2019 and TEST ON 2009 and 2010
# X_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2008-09-30 00:00:00"], 
#                       df_ML.loc["2012-10-01 00:00:00":"2019-09-30 00:00:00"]]).drop(['ROS'], axis = 1) #X = features for training years
# y_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2008-09-30 00:00:00"], 
#                       df_ML.loc["2012-10-01 00:00:00":"2019-09-30 00:00:00"]]) #y = target for training years
# y_train = pd.Series(y_train.ROS)
# X_test = df_ML["2008-10-01 00:00:00":"2010-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
# y_test = df_ML.loc["2008-10-01 00:00:00":"2010-09-30 00:00:00", 'ROS'] #y = target for test years

# ###LEARN FROM 2006-2009 and 2014-2019 and 2016-2019 and TEST ON 2010 and 2013
# X_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2009-09-30 00:00:00"], 
#                       df_ML.loc["2013-10-01 00:00:00":"2019-09-30 00:00:00"]]).drop(['ROS'], axis = 1) #X = features for training years
# y_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2009-09-30 00:00:00"], 
#                       df_ML.loc["2013-10-01 00:00:00":"2019-09-30 00:00:00"]]) #y = target for training years
# y_train = pd.Series(y_train.ROS)
# X_test = df_ML["2009-10-01 00:00:00":"2013-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
# y_test = df_ML.loc["2009-10-01 00:00:00":"2013-09-30 00:00:00", 'ROS'] #y = target for test years

# ###LEARN FROM 2006-2010 and 2015-2019 and 2016-2019 and TEST ON 2013 and 2014
# X_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2010-09-30 00:00:00"], 
#                       df_ML.loc["2014-10-01 00:00:00":"2019-09-30 00:00:00"]]).drop(['ROS'], axis = 1) #X = features for training years
# y_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2010-09-30 00:00:00"], 
#                       df_ML.loc["2014-10-01 00:00:00":"2019-09-30 00:00:00"]]) #y = target for training years
# y_train = pd.Series(y_train.ROS)
# X_test = df_ML["2012-10-01 00:00:00":"2014-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
# y_test = df_ML.loc["2012-10-01 00:00:00":"2014-09-30 00:00:00", 'ROS'] #y = target for test years

# ###LEARN FROM 2006-2013 and 2016-2019 and TEST ON 2014 and 2015
# X_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2013-09-30 00:00:00"], 
#                       df_ML.loc["2015-10-01 00:00:00":"2019-09-30 00:00:00"]]).drop(['ROS'], axis = 1) #X = features for training years
# y_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2013-09-30 00:00:00"], 
#                       df_ML.loc["2015-10-01 00:00:00":"2019-09-30 00:00:00"]]) #y = target for training years
# y_train = pd.Series(y_train.ROS)
# X_test = df_ML["2013-10-01 00:00:00":"2015-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
# y_test = df_ML.loc["2013-10-01 00:00:00":"2015-09-30 00:00:00", 'ROS'] #y = target for test years

# ###LEARN FROM 2006-2014 and 2017-2019 and TEST ON 2015 and 2016
# X_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2014-09-30 00:00:00"], 
#                       df_ML.loc["2016-10-01 00:00:00":"2019-09-30 00:00:00"]]).drop(['ROS'], axis = 1) #X = features for training years
# y_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2014-09-30 00:00:00"], 
#                       df_ML.loc["2016-10-01 00:00:00":"2019-09-30 00:00:00"]]) #y = target for training years
# y_train = pd.Series(y_train.ROS)
# X_test = df_ML["2014-10-01 00:00:00":"2016-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
# y_test = df_ML.loc["2014-10-01 00:00:00":"2016-09-30 00:00:00", 'ROS'] #y = target for test years

# ###LEARN FROM 2006-2015 and 2018-2019 and TEST ON 2016 and 2017
# X_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2015-09-30 00:00:00"], 
#                       df_ML.loc["2017-10-01 00:00:00":"2019-09-30 00:00:00"]]).drop(['ROS'], axis = 1) #X = features for training years
# y_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2015-09-30 00:00:00"], 
#                       df_ML.loc["2017-10-01 00:00:00":"2019-09-30 00:00:00"]]) #y = target for training years
# y_train = pd.Series(y_train.ROS)
# X_test = df_ML["2015-10-01 00:00:00":"2017-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
# y_test = df_ML.loc["2015-10-01 00:00:00":"2017-09-30 00:00:00", 'ROS'] #y = target for test years

# # ###LEARN FROM 2006-2016 and 2019 and TEST ON 2017 and 2018
# X_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2016-09-30 00:00:00"], 
#                       df_ML.loc["2018-10-01 00:00:00":"2019-09-30 00:00:00"]]).drop(['ROS'], axis = 1) #X = features for training years
# y_train = pd.concat([df_ML.loc["2005-10-01 00:00:00":"2016-09-30 00:00:00"], 
#                       df_ML.loc["2018-10-01 00:00:00":"2019-09-30 00:00:00"]]) #y = target for training years
# y_train = pd.Series(y_train.ROS)
# X_test = df_ML["2016-10-01 00:00:00":"2018-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
# y_test = df_ML.loc["2016-10-01 00:00:00":"2018-09-30 00:00:00", 'ROS'] #y = target for test years

# ###LEARN FROM 2006-2017 and TEST ON 2018-2019
# X_train = df_ML.loc["2005-10-01 00:00:00":"2017-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for training years
# y_train = df_ML.loc["2005-10-01 00:00:00":"2017-09-30 00:00:00", 'ROS'] #y = target for training years
# X_test = df_ML["2017-10-01 00:00:00":"2019-09-30 00:00:00"].drop(['ROS'], axis = 1) #X = features for test years
# y_test = df_ML.loc["2017-10-01 00:00:00":"2019-09-30 00:00:00", 'ROS'] #y = target for test years

#%% FEATURE COLUMNS

###Create feature columns - print to list, copy/paste and remove the target variable
feature_cols = X_train.columns.tolist()
# print(feature_cols)

#%% Run Decision Tree Classifier   
maxdepth = 4

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=maxdepth, criterion='gini', splitter='best', max_features=None)

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
fig = plt.figure(figsize=(4, 3), dpi = 300)
sns.heatmap(confusion_matrix(y_test, y_pred), fmt=".0f", annot=True)
plt.xlabel("Actual", size=12)
plt.ylabel("Predicted", size=12)
plt.title("a. Model Performance", size=14)

from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(4, 3), dpi = 300)
sns.heatmap(confusion_matrix(y_train, y_pred_train), fmt=".0f", annot=True)
plt.xlabel("Predicted", size=12)
plt.ylabel("Actual", size=12)
plt.title("b. Test Performance", size=14)

#%% VISUALIZE DECISION TREE

dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=feature_cols,
                      max_depth=maxdepth,
                      class_names=['warm_day', 'ROS'],  
                      filled=True, rounded=True,  
                      special_characters=False, fontname= 'sans')  
graph = graphviz.Source(dot_data)  
graph.format = 'pdf'
graph.render('dtree_clean.pdf',view=True)


#%% get importance
importance = clf.feature_importances_
indices = np.argsort(importance)

fig, axes = plt.subplots(1, 1, figsize=(12, 14))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
