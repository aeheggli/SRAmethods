# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 08:30:19 2021

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

#update SNOTEL station id & water year
stationid = '428'
year= '2017'

# QA & QC Definitions

# Quality Control flags included:
#
# Flag    Name                Description
#  V      Valid               Validated Data
#  E      Edit                Edit, minor adjustment for sensor noise
#  S      Suspect             Suspect data
# 
# Quality Assurance flags included:
#
# Flag    Name                Description
#  R      Raw                 No Human Review
#  F      Flagged	          Automated Data Flag
#  P      Provisional         Preliminary Human Review
#  A      Approved            Processing and Final Review Completed


#%% Read reference daily data

#update working directory accordingly
os.chdir('/Users/anne/OneDrive/Data/qc_for_SRA/raw')

#read daily SNOTEL data
snotel_daily = pd.read_csv('sd_428_WY' + year + '.csv', parse_dates=['Date']) #data is start of day values
#name columns with unique sd (snotel daily) identifier
snotel_daily.columns =['date','sd_temp_avg_C','sd_temp_avg_qc','sd_temp_max_C','sd_temp_max_qc','sd_temp_min_C','sd_temp_min_qc','sd_temp_obs_C','sd_temp_obs_qc', 'sd_precip_accum_mm', 'sd_precip_accum_qc', 'sd_precip_24hr_mm','sd_precip_24hr_qc', 'sd_precip_24hrsnowadj_mm','sd_precip_24hradj_qc','sd_depth_cm', 'sd_depth_qc', 'sd_SWE_mm', 'sd_SWE_qc','sd_sm2_%','sd_sm2_qc','sd_sm8_%','sd_sm8_qc','sd_sm20_%','sd_sm20_qc','sd_sm2_avg_%','sd_sm2_avg_qc','sd_sm8_avg_%','sd_sm8_avg_qc','sd_sm20_avg_%','sd_sm20_avg_qc','sd_sm2_max_%','sd_sm2_max_qc','sd_sm8_max_%','sd_sm8_max_qc','sd_sm20_max_%','sd_sm20_max_qc', 'sd_sm2_min_%','sd_sm2_min_qc','sd_sm8_min_%','sd_sm8_min_qc','sd_sm20_min_%','sd_sm20_min_qc','sd_st2_avg_%','sd_st2_avg_qc','sd_st8_avg_%','sd_st8_avg_qc','sd_st20_avg_%','sd_st20_avg_qc','sd_st2_max_%','sd_st2_max_qc','sd_st8_max_%','sd_st8_max_qc','sd_st20_max_%','sd_st20_max_qc','sd_st2_min_%','sd_st2_min_qc','sd_st8_min_%','sd_st8_min_qc','sd_st20_min_%','sd_st20_min_qc','sd_st2_%','sd_st2_qc','sd_st8_%','sd_st8_qc','sd_st20_%','sd_st20_qc','sd_density_%','sd_density_qc', 'sd_snowrainratio','sd_snowrainratio_qc']
snotel_daily['date'] = snotel_daily['date'].apply(lambda x: pd.Timestamp(x).replace(hour=0, minute=0, second=0))

#read CSSL manual observations (if availabele)
obs = pd.read_csv('obs_' + year + '.csv', parse_dates=['date'])
#name columns with unique obs (observed) identifier
obs.columns = ['date','year','obs_tempmax_C','obs_tempmin_C','obs_precip_24hrmm','obs_precip_cum_mm','obs_precip_%snow','obs_precip_%rain','obs_snowhn24_cm','obs_snowtotal_cm','obs_HS_cm','obs_SWE_cm','obs_SWE_mm']

#%% WORKING DATA FOR MANUAL QC REVIEW

#READ DATA WORKING DATA FILE
os.chdir('/Users/anne/OneDrive/Data/qc_for_SRA')
prelim = pd.read_csv('pubication_test/428_WY'+year+'_prelim.csv', parse_dates=['date'])
prelim['density'] = prelim['sh_SWE_manual_mm']/prelim['sh_depth_manual_cm']*10

#PLOT DATA
fig = go.Figure()

#SWE
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_SWE_mm'], line=dict(color='#8F929F'), name='SNOTEL SWE', yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['sh_SWE_auto_mm'], line=dict(color='#0000FF'), name='sh_SWE_auto6h_mm', yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['sh_SWE_auto24h_mm'], line=dict(color='deeppink'), name='sh_SWE_auto24h_mm', yaxis='y1'))
fig.add_trace(go.Scatter(mode = 'markers', x=snotel_daily['date'], y=snotel_daily['sd_SWE_mm'], line=dict(color='#3B9212'), name='snotel_daily', yaxis='y1'))
fig.add_trace(go.Scatter(mode = 'markers', x=obs['date'], y=obs['obs_SWE_mm'], line=dict(color='#EC0051'), name='CSSL Obs SWE', yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['sh_SWE_manual_mm'], line=dict(color='#000000'), name='sh_SWE_manual_mm', yaxis='y1'))

#Depth
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['sh_snowdepth_cm'],line=dict(color='#8F929F'),name='SNOTEL depth',yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['sh_depth_auto_cm'],line=dict(color='#0000FF'),name='sh_depth_auto_cm',yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['sh_depth_auto12h_cm'],line=dict(color='#00D1D1'),name='sh_depth_auto12h_cm',yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['sh_depth_auto24h_cm'],line=dict(color='#00008B'),name='sh_depth_auto24h_cm',yaxis='y1'))
fig.add_trace(go.Scatter(mode = 'markers', x=snotel_daily['date'], y=snotel_daily['sd_depth_cm'],  line=dict(color='#3B9212'), name='SNOTEL Daily Depth',yaxis='y1'))
fig.add_trace(go.Scatter(mode = 'markers', x=obs['date'], y=obs['obs_HS_cm'],  line=dict(color='#EC0051'), name='CSSL Obs Depth', yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['sh_depth_manual_cm'], line=dict(color='#000000'),name='sh_depthmanual_cm',yaxis='y1'))

#Density
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['density'], name='Density',opacity=0.8, yaxis='y2'))
fig.add_trace(go.Scatter(mode = 'markers', x=snotel_daily['date'],y=snotel_daily['sd_density_%'], line=dict(color='#3B9212'), name='sd_density_%', yaxis='y1'))

#Precip
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['sh_precip_mm'], line=dict(color='#8F929F'), name='sh_precip_mm', yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['sh_precip_auto_mm'], line=dict(color='#0000FF'), name='sh_precipcumqc_mm', yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['w10_precip_auto'], line=dict(color='#880ED4'), name='w10_precip_auto', yaxis='y1'))
fig.add_trace(go.Scatter(mode = 'markers', x=snotel_daily['date'],y=snotel_daily['sd_precip_accum_mm'], line=dict(color='#3B9212'), name='sd_precip_accum_mm', yaxis='y1'))
fig.add_trace(go.Scatter(mode = 'markers', x=obs['date'], y=obs['obs_precip_cum_mm'], line=dict(color='#EC0051'), name='CSSL Obs Precip', yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'], y=prelim['sh_precip_manual_mm'], line=dict(color='#000000'), name='sh_precip_manual_mm',yaxis='y1'))

#Temp
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_temp_C'], line=dict(color='#8F929F'),name='sh_temp_C',opacity=0.7, yaxis='y2'))
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_tempcorrected_C'], line=dict(color='#0000FF'), name='sh_tempcorrected_C',opacity=0.7, yaxis='y2'))

#Soil Moistrue
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_sm2_%'], line=dict(color='#8F929F'),name='sh_sm2_pct', yaxis='y2'))
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_sm8_%'], line=dict(color='#8F929F'),name='sh_sm8_pct', yaxis='y2'))
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_sm20_%'], line=dict(color='#8F929F'),name='sh_sm20_pct', yaxis='y2'))
prelim['sh_sm2_auto'] = prelim['sh_sm2_%'].rolling(6, center=True, closed='right').median()
prelim['sh_sm8_auto'] = prelim['sh_sm8_%'].rolling(6, center=True, closed='right').median()
prelim['sh_sm20_auto'] = prelim['sh_sm20_%'].rolling(6, center=True, closed='right').median()
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_sm2_auto'], name='sh_sm2_auto_', yaxis='y2'))
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_sm8_auto'], name='sh_sm8_auto_', yaxis='y2'))
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_sm20_auto'],name='sh_sm20_pct', yaxis='y2'))

#Soil Temperature
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_st2_C'], line=dict(color='#8F929F'),name='sh_st2_C', yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_st8_C'], line=dict(color='#8F929F'),name='sh_st8_C', yaxis='y1'))
fig.add_trace(go.Scatter(x=prelim['date'],y=prelim['sh_st20_C'], line=dict(color='#8F929F'),name='sh_st20_C', yaxis='y1'))

fig.update_layout(
    title='Central Sierra Snow Lab',
    yaxis=dict(title='SWE (mm) / Depth (cm)'),
    yaxis2=dict(title='Precip (mm)/Temp (C)', overlaying='y', side='right'),
    legend_title='Legend',
    plot_bgcolor = '#FBFBFB',
)
fig.show()

#%% EXPORT PROVISIONAL DATA PRODUCT
year= '2017'
stationid = '428'
os.chdir('/Users/anne/OneDrive/Data/qc_for_SRA/pubication_test')
provisional = pd.read_csv(stationid+'_WY'+year+'_prelim.csv', parse_dates=['date'])

#update density calculation
provisional['density'] = provisional['sh_SWE_manual_mm']/provisional['sh_depth_manual_cm']*10

#Verify QC Flag for data
provisional['sh_SWE_manual_qcflag'] = np.where((provisional['sh_SWE_mm'] == provisional['sh_SWE_manual_mm']) & (provisional['sh_SWE_manual_qcflag']!="S"), provisional['sh_SWE_qc'], provisional['sh_SWE_manual_qcflag'])
provisional['sh_depth_manual_qcflag'] = np.where((provisional['sh_snowdepth_cm'] == provisional['sh_depth_manual_cm']) & (provisional['sh_depth_manual_qcflag']!="S"), provisional['sh_snowdepth_qc'], provisional['sh_depth_manual_qcflag'])
provisional['sh_precip_manual_qcflag'] = np.where((provisional['sh_precip_mm'] == provisional['sh_precip_manual_mm']) & (provisional['sh_precip_manual_qcflag']!="S"), provisional['sh_precip_qc'], provisional['sh_precip_manual_qcflag'])

path = '/Users/anne/OneDrive/Data/QC_for_SRA/pubication_test/'+stationid+'_WY'+year+'_provisional.csv'
provisional.to_csv(path)

#%% EXPORT FINAL DATA FOR ML

year= '2017'
stationid = '428'
os.chdir('/Users/anne/OneDrive/Data/qc_for_SRA/pubication_test')
ML_export = provisional.set_index('date')
ML_export = provisional.drop(['sh_SWE_mm', 'sh_SWE_qc', 'sh_SWE_auto24h_mm', 'sh_SWE_auto_mm', 'sh_SWE_auto_qcflag', 'sh_SWE_1hrdiff', 'sh_snowdepth_cm', 'sh_snowdepth_qc', 'sh_depth_auto_cm',
 'sh_depth_auto12h_cm', 'sh_depth_auto24h_cm', 'sh_depth_auto_qcflag', 'density', 'sh_precip_mm', 'sh_precip_qc', 'sh_precip_auto_mm', 'sh_precip_auto_qcflag', 'sh_precip_1hrdiff','w10_precip_1hrdiff', 'w10_precip_auto',
 'w10_precip_auto_qcflag', 'sh_temp_C', 'sh_temp_qc'], axis=1) #name columns
# final.columns = ['date', 'SWE_mm', 'depth_cm', 's_precip_mm','w_precip_mm', 'w_temp_C', 'sm2_pct','sm8_pct', 'sm20_pct','s_precip_1htot']

MLpath = '/Users/anne/OneDrive/Data/QC_for_SRA/pubication_test/'+stationid+'_WY'+year+'_ML.csv' #puts in the general folder to avoid accidentally overwriting final data. This file must be manually moved to the 'final' folder and also to the 'ML test' folder
ML_export.to_csv(MLpath)
