# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:36:05 2021

@author: anne
"""
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pio.renderers.default='browser'

######### SET PLOTTING DEFAULTS ###########################
# palette = sns.color_palette("colorblind")
# sns.palplot(palette)
palette = "colorblind"
ROScolors = ["#0173B2", "#029E73"]
ROSpalette = sns.set_palette(sns.color_palette(ROScolors))
plt.rcParams['lines.linewidth'] = 1

SMALL_SIZE = 8
MEDIUM_SIZE = 8
LARGE_SIZE = 8

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title


#%%IMPORT DATA 

exec(open('/Users/anne/OneDrive/Data/Python/CSSLread_clean.py').read())
df = pd.DataFrame(clean)
df['water_year'] = df.date.dt.year.where(df.date.dt.month < 10, df.date.dt.year + 1)

### DAILY DATA DOWNLOADED FROM NRCS
exec(open('/Users/anne/OneDrive/Data/Python/CSSLread_sd.py').read())
sd = pd.DataFrame(sd)

exec(open('/Users/anne/OneDrive/Data/Python/CSSLread_sh.py').read())
sh = pd.DataFrame(sh)

#%% FEATURES

##PRECIPITATION WINDOWS
df['precip_1h'] = df['sh_precip_manual_mm'].diff()
df['precip_6h'] = df['precip_1h'].rolling(6).sum()

# SWE CHANGE
df['delta_SWE_1h'] = df['sh_SWE_manual_mm'].diff()
df['delta_SWE_2h'] = df['sh_SWE_manual_mm'].diff(2)
df['delta_SWE_3h'] = df['sh_SWE_manual_mm'].diff(3)
df['delta_SWE_4h'] = df['sh_SWE_manual_mm'].diff(4)
df['delta_SWE_5h'] = df['sh_SWE_manual_mm'].diff(5)
df['delta_SWE_6h'] = df['sh_SWE_manual_mm'].diff(6)
df['delta_SWE_12h'] = df['sh_SWE_manual_mm'].diff(12)
df['delta_SWE_24h'] = df['sh_SWE_manual_mm'].diff(24)

##DENSITY WINDOWS
df['density_pct']=df['sh_SWE_manual_mm']/df['sh_depth_manual_cm']*10
df['density_1h_earlier'] = df['density_pct'].shift(1)

##TEMPERATURE WINDOWS
df['temp_6h_max'] = df['sh_tempcorrected_C'].rolling(6).max()

df['temp_6h_min'] = df['sh_tempcorrected_C'].rolling(6).min()

##SOIL MOISTURE
df['sh_sm2_auto_%'] = df['sh_sm2_%'].rolling(6, center=True, closed='right').median()
df['sh_sm8_auto_%'] = df['sh_sm8_%'].rolling(6, center=True, closed='right').median()
df['sh_sm20_auto_%'] = df['sh_sm20_%'].rolling(6, center=True, closed='right').median()

##SOIL MOISTURE CHANGE
df['delta_sm2'] = df['sh_sm2_%'].diff()
df['delta_sm2_2h'] = df['sh_sm2_%'].diff(2)
df['delta_sm8'] = df['sh_sm8_%'].diff()
df['delta_sm8_2h'] = df['sh_sm8_%'].diff(2)
df['delta_sm20'] = df['sh_sm20_%'].diff()
df['delta_sm20_2h'] = df['sh_sm20_%'].diff(2)

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
df['water_release_target'] = np.where((df['water_release1'] == 1) | (df['water_release2'] == 1) |(df['saturation'] == 1), 1,0)

#%% FILTER DATA & CREATE TARGET
df = df.set_index('date')
df_ML = pd.concat([
                    # df.loc["2005-10-01 00:00:00":"2006-04-18 00:00:00"],
                    # df.loc["2006-10-01 00:00:00":"2007-03-01 00:00:00"], #Defined by peak SWE
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
# df = df.loc[(df['w10_precip_manual_qcflag']!= "S")]

# FILTER FOR SNOW COVER 
df_ML = df_ML.loc[(df['sh_SWE_manual_mm']>=100)]
df_ML = df_ML.loc[(df['water_release_target']==1)]

# ADD TARGET VARIABLES: ROS & WARM DAY MELT MANUALLY IDENTIFIED
target = pd.read_csv("/Users/anne/OneDrive/Data/QC_for_SRA/MLdata/df_ML_ROSidentified.csv", parse_dates=['date'], index_col=['date'])
df_ML = pd.concat([df_ML, target], axis=1)

df_ML['unidentified'] = np.where((df_ML['warmday']==0) & (df_ML['ROS']==0), 1, 0)
df_ML = df_ML[df_ML.unidentified == 0]

print(df_ML.groupby('classification').count()) 
df_ML = df_ML.drop(['unidentified'], axis=1)
  
df_MLsummary = df_ML.describe()
#%% DROP COLUMNS 
df_ML = df_ML.drop(['sh_SWE_manual_qcflag','sh_SWE_manual_qaflag','sh_depth_manual_qcflag','sh_depth_manual_qaflag','sh_precip_manual_qcflag',
                    'sh_precip_manual_qaflag','w10_precip_manual_qcflag', 'w10_precip_manual_qaflag','sh_temp_qcflag','sh_temp_qaflag',
                    'sh_sm2_%','sh_sm2_qc','sh_sm8_%','sh_sm8_qc','sh_sm20_%','sh_sm20_qc','sh_sm2_C_qaflag','sh_sm8_C_qaflag','sh_sm20_C_qaflag',
                    'sh_st2_C','sh_st2_qc','sh_st8_C','sh_st2_qc.1','sh_st20_C','sh_st20_qc','sh_st2_C_qaflag','sh_st8_C_qaflag','sh_st20_C_qaflag','sh_sm2_auto_%','sh_sm8_auto_%','sh_sm20_auto_%',
                    'delta_sm2','delta_sm2_2h','delta_sm8','delta_sm8_2h', 'delta_sm20', 'delta_sm20_2h', 'delta_autosm2','delta_autosm2_2h','delta_autosm8','delta_autosm8_2h', 'delta_autosm20', 'delta_autosm20_2h',
                    'water_release1','water_release2','saturation'], axis=1)

# df_ML = df_ML.fillna(-9999)
df_ML['ROS + SWE loss'] = np.where((df_ML['delta_SWE_1h']<=-2), 1, 0)
df_ML['classification'] = np.where((df_ML['ROS + SWE loss']==1), "ROS + SWE loss", df_ML['classification'])
df_ROS = df_ML.loc[(df_ML['classification']=='ROS') | (df_ML['classification']=='ROS + SWE loss')]

print(df_ML.groupby('classification').count()) 
#%%
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
ROSclassification = df_ML[df_ML.classification == 'ROS'].describe()
warmclassification = df_ML[df_ML.classification == 'warmday'].describe()
ROSSWElossclassification = df_ML[df_ML.classification == 'ROS + SWE loss'].describe()

#%% Density Histogram
palette = sns.color_palette('colorblind', n_colors=3)
#    colorblind=["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"],
fig = plt.figure(figsize=(6.25, 3.5), dpi = 300)
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, :-1])
sns.histplot(data=df_ML, x="density_1h_earlier", hue="classification", palette = ['#F26739','#BF2026','#EAA2DC'], multiple="dodge", stat='count', shrink=0.9, bins=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55], kde=True, ax=ax1)
plt.xlabel('Density 1-hour earlier (%)', fontsize=8)
plt.ylabel('Count', fontsize=8)
ax1.set_xticks([10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
ax1.set_xticklabels(('10', '15', '20', '25', '30', '35', '40', '45', '50', '55'))
ax1.get_legend().remove()

ax2 = fig.add_subplot(gs[:, -1])
sns.violinplot(data=df_ML, x= 'classification', y="density_1h_earlier", inner='quartile', width=0.5, palette = ['#F26739','#BF2026','#EAA2DC'], ax=ax2, cut=0)
plt.ylabel('Density 1-hour earlier (%)', fontsize=8)
ax2.set_xlabel(None)

# plt.savefig("/Users/anne/OneDrive/Data/QC_for_SRA/Plots/Frequency_Density.pdf", transparent=False)

#%% Temp Histogram
palette = sns.color_palette('colorblind', n_colors=2)

fig = plt.figure(figsize=(6.25, 3.5), dpi = 300)
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, :-1])
sns.histplot(data=df_ML, x="temp_6h_max", hue="classification", multiple="dodge", palette = ['#F26739','#BF2026','#EAA2DC'], shrink=0.9, bins=[-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], kde=True, ax=ax1)
plt.xlabel('6-hour Maximum Temperature (C)', fontsize=8)
plt.ylabel('Count', fontsize=12)
ax1.set_xticks([-4,-2,0,2,4,6,8,10,12,14])
ax1.set_xticklabels(('-4','-2','0','2','4','6','8','10','12','14'))
ax1.get_legend().remove()

ax2 = fig.add_subplot(gs[:, -1])
sns.violinplot(data=df_ML, x= 'classification', y="temp_6h_max", inner='quartile', width=0.5, palette = ['#F26739','#BF2026','#EAA2DC'], ax=ax2, cut=0)
plt.ylabel('6-hour Maximum Temperature (C)', fontsize=8)
ax2.set_xlabel(None)

# plt.savefig("/Users/anne/OneDrive/Data/QC_for_SRA/Plots/Frequency_Temp.pdf", transparent=False)

#%% ROS Precip Histogram
fig = plt.figure(figsize=(6.25, 3.5), dpi = 300)
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, :-1])
sns.histplot(data=df_ROS, x="precip_6h", hue = 'classification', multiple="dodge", palette = ['#F26739','#BF2026'], shrink=0.9, kde=True, ax=ax1)
plt.xlabel('6-hour Precip (mm)', fontsize=8)
plt.ylabel('Count', fontsize=8)
ax1.set_xticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65])
ax1.set_xticklabels(('0','5','10','15','20','25','30','35','40','45','50','55','60','65'))
ax1.get_legend().remove()

ax2 = fig.add_subplot(gs[:, -1])
sns.violinplot(data=df_ROS, x= 'classification', y="precip_6h", inner='quartile', width=0.5, palette = ['#F26739','#BF2026'], ax=ax2, cut=0)
plt.ylabel('6-hour Precipitation (mm)', fontsize=8)
ax2.set_xlabel(None)

# plt.savefig("/Users/anne/OneDrive/Data/QC_for_SRA/Plots/Frequency_Precip.pdf", transparent=False)

#%% Temp Histogram (MIN)
palette = sns.color_palette('colorblind', n_colors=2)

fig = plt.figure(figsize=(6.25, 3.5), dpi = 300)
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, :-1])
sns.histplot(data=df_ML, x="temp_6h_min", hue="classification", multiple="dodge", palette = ['#0173b2','#029e73','#de8f05'], shrink=0.9, bins=[-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], kde=True, ax=ax1)
plt.xlabel('6-hour Minimum Temperature (C)', fontsize=8)
plt.ylabel('Count', fontsize=8)
ax1.set_xticks([-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14])
ax1.set_xticklabels(('-14','-12','-10','-8','-6','-4','-2','0','2','4','6','8','10','12','14'))

ax2 = fig.add_subplot(gs[:, -1])
sns.violinplot(data=df_ML, x= 'classification', y="temp_6h_min", inner='quartile', width=0.5, palette = ['#0173b2','#029e73','#de8f05'], ax=ax2, cut=0)
plt.ylabel('6-hour Minumum Temperature (C)', fontsize=8)
ax2.set_xlabel(None)

# plt.savefig("/Users/anne/OneDrive/Data/QC_for_SRA/Plots/Frequency_TempMin.pdf", transparent=False)

#%% SWE change
fig = plt.figure(constrained_layout=True, figsize=(9, 6), dpi = 300)
gs = fig.add_gridspec(1, 5)
ax1 = fig.add_subplot(gs[0, 0])
sns.violinplot(data=df_ROS,  y="delta_SWE_1h", ax=ax1,inner='quartile', color='#31a354', alpha=0.5, cut=0)
ax1.axhline(0, color="gray", zorder=0) 
plt.ylabel(r'$\Delta$ SWE (mm)', fontsize=16)
plt.ylim(-120,140)
plt.xlabel('1-hour')

ax2 = fig.add_subplot(gs[0, 1])
sns.violinplot(data=df_ROS,  y="delta_SWE_3h", inner='quartile',color='#31a354', ax=ax2, cut=0)
ax2.axhline(0, color="gray", zorder=0) 
plt.ylabel(None)
plt.ylim(-120,140)
plt.xlabel('3-hours')

ax3 = fig.add_subplot(gs[0, 2])
sns.violinplot(data=df_ROS,  y="delta_SWE_6h", inner='quartile',color='#31a354', ax=ax3, cut=0)
ax3.axhline(0, color="gray", zorder=0) 
plt.ylabel(None)
plt.ylim(-120,140)
plt.xlabel('6-hours')

ax4 = fig.add_subplot(gs[0, 3])
sns.violinplot(data=df_ROS,  y="delta_SWE_12h", inner='quartile',color='#31a354', ax=ax4, cut=0)
ax4.axhline(0, color="gray", zorder=0) 
plt.ylabel(None)
plt.ylim(-120,140)
plt.xlabel('12-hours')

ax5 = fig.add_subplot(gs[0, 4])
sns.violinplot(data=df_ROS,  y="delta_SWE_24h", inner='quartile',color='#31a354', ax=ax5, cut=0)
ax5.axhline(0, color="gray", zorder=0) 
plt.ylabel(None)
plt.ylim(-120,140)
plt.xlabel('24-hours')

fig.suptitle('a.', fontsize=16, x=0.0, y=1.05)
