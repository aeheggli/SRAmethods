# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:12:01 2021

@author: anne
"""
import os
import pandas as pd 
import copy
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
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
# 

#%% Read all data
#update working directory accordingly
os.chdir('/Users/anne/OneDrive/Data/qc_for_SRA/raw')

#read hourly SNOTEL data
snotel_hourly = pd.read_csv('sh_428_WY'+year+'.csv', parse_dates=['Date'])
#name columns with unique sh (snotel hourly) identifier
snotel_hourly.columns = ['date','sh_temp_C','sh_temp_qc','sh_precip_mm','sh_precip_qc','sh_snowdepth_cm','sh_snowdepth_qc','sh_SWE_mm','sh_SWE_qc','sh_sm2_%','sh_sm2_qc','sh_sm8_%','sh_sm8_qc','sh_sm20_%','sh_sm20_qc','sh_st2_C','sh_st2_qc','sh_st8_C','sh_st2_qc','sh_st20_C','sh_st20_qc'] 

#read daily SNOTEL data
snotel_daily = pd.read_csv('sd_428_WY' + year + '.csv', parse_dates=['Date']) #data is start of day values
#name columns with unique sd (snotel daily) identifier
snotel_daily.columns =['date','sd_temp_avg_C','sd_temp_avg_qc','sd_temp_max_C','sd_temp_max_qc','sd_temp_min_C','sd_temp_min_qc','sd_temp_obs_C','sd_temp_obs_qc', 'sd_precip_accum_mm', 'sd_precip_accum_qc', 'sd_precip_24hr_mm','sd_precip_24hr_qc', 'sd_precip_24hrsnowadj_mm','sd_precip_24hradj_qc','sd_depth_cm', 'sd_depth_qc', 'sd_SWE_mm', 'sd_SWE_qc','sd_sm2_%','sd_sm2_qc','sd_sm8_%','sd_sm8_qc','sd_sm20_%','sd_sm20_qc','sd_sm2_avg_%','sd_sm2_avg_qc','sd_sm8_avg_%','sd_sm8_avg_qc','sd_sm20_avg_%','sd_sm20_avg_qc','sd_sm2_max_%','sd_sm2_max_qc','sd_sm8_max_%','sd_sm8_max_qc','sd_sm20_max_%','sd_sm20_max_qc', 'sd_sm2_min_%','sd_sm2_min_qc','sd_sm8_min_%','sd_sm8_min_qc','sd_sm20_min_%','sd_sm20_min_qc','sd_st2_avg_%','sd_st2_avg_qc','sd_st8_avg_%','sd_st8_avg_qc','sd_st20_avg_%','sd_st20_avg_qc','sd_st2_max_%','sd_st2_max_qc','sd_st8_max_%','sd_st8_max_qc','sd_st20_max_%','sd_st20_max_qc','sd_st2_min_%','sd_st2_min_qc','sd_st8_min_%','sd_st8_min_qc','sd_st20_min_%','sd_st20_min_qc','sd_st2_%','sd_st2_qc','sd_st8_%','sd_st8_qc','sd_st20_%','sd_st20_qc','sd_density_%','sd_density_qc', 'sd_snowrainratio','sd_snowrainratio_qc']
snotel_daily['date'] = snotel_daily['date'].apply(lambda x: pd.Timestamp(x).replace(hour=0, minute=0, second=0))

#read CSSL manual observations (if availabele)
obs = pd.read_csv('obs_' + year + '.csv', parse_dates=['date'])
#name columns with unique obs (observed) identifier
obs.columns = ['date','year','obs_tempmax_C','obs_tempmin_C','obs_precip_24hrmm','obs_precip_cum_mm','obs_precip_%snow','obs_precip_%rain','obs_snowhn24_cm','obs_snowtotal_cm','obs_HS_cm','obs_SWE_cm','obs_SWE_mm']
 
#read WRCC 10-minute data
wrcc = pd.read_csv('wrcc_cssl_WY' + year + '.csv', header=2, parse_dates=['date'])
#name columns with unique w10 (WRCC 10-minute) identifier
wrcc.columns = ['date','w10_solarrad','w10_solarreflect','w10_unknown1','w10_unknown2','w10_unknown3','w10_geonorfeqhz','w10_geonorstd','w10_geonorcm','w10_mxtempF','w10_mntempF','w10_avtempF','w10_mxrh%','w10_mnrh%','w10_rh%','w10_bpinhg','w10_solarrad2','w10_snowdepthin','w10_mxsnowdepthin','w10_minsnowdepthin','w10_snowstd','w10_SWEin']

#%% Level 1 - range check

# Establish dynamic range check values from NRCS SNOTEL profile
#read CSSL profiles
CSSL_profiles = pd.read_csv('CSSL_profilesmetric.csv', parse_dates=['date'])
dateoffset = int(year) - 2006
CSSL_profiles['date'] = CSSL_profiles['date'] +  pd.offsets.DateOffset(years=dateoffset)
CSSL_profiles = CSSL_profiles.set_index('date')
snotel_hourly = snotel_hourly.set_index('date')

#Apply daily profile to each time stamp for automated QC.
snotel_hourly = pd.concat([snotel_hourly, CSSL_profiles], axis=1)
snotel_hourly['depth_lower'] = snotel_hourly['depth_lower'].fillna(method = 'ffill')
snotel_hourly['depth_upper'] = snotel_hourly['depth_upper'].fillna(method = 'ffill')
snotel_hourly['depth_decrease'] = snotel_hourly['depth_decrease'].fillna(method = 'ffill')
snotel_hourly['depth_increase'] = snotel_hourly['depth_increase'].fillna(method = 'ffill')
snotel_hourly['SWE_lower'] = snotel_hourly['SWE_lower'].fillna(method = 'ffill')
snotel_hourly['SWE_upper'] = snotel_hourly['SWE_upper'].fillna(method = 'ffill')
snotel_hourly['SWE_decrease'] = snotel_hourly['SWE_decrease'].fillna(method = 'ffill')
snotel_hourly['SWE_increase'] = snotel_hourly['SWE_increase'].fillna(method = 'ffill')
snotel_hourly['prec_lower'] = snotel_hourly['prec_lower'].fillna(method = 'ffill')
snotel_hourly['prec_upper'] = snotel_hourly['prec_upper'].fillna(method = 'ffill')
snotel_hourly['prec_decrease'] = snotel_hourly['prec_decrease'].fillna(method = 'ffill')
snotel_hourly['prec_increase'] = snotel_hourly['prec_increase'].fillna(method = 'ffill')
# 'depth_lower','depth_upper','depth_decrease','depth_increase','SWE_lower','SWE_upper','SWE_decrease','SWE_increase','prec_lower','prec_upper','prec_decrease','prec_increase'
snotel_hourly = snotel_hourly.reset_index()

#Absolute value range check
SWEmax = 1836  #mm
SWEmin = 0  #mm
depthmax = 564 #cm
depthmin = 0 #cm
precipseasonmax = 3297 #mm
precipseasonmin = 0 #mm
precip24hrmax = 124 #mm/hr
precip24hrmin = 0 #mm/hr
precip1hrmax = 10 #mm/hr
precip1hrmin = 0 #mm/hr
tempmax = 30.4 #C
tempmin = -20 #C

#%% AUTOMATED QC ROUTINE

#### SWE ####
qc_SWE = copy.deepcopy(snotel_hourly) 
qc_SWE = qc_SWE.drop(['sh_temp_C','sh_temp_qc','sh_precip_mm','sh_precip_qc','sh_snowdepth_cm','sh_snowdepth_qc','sh_sm2_%','sh_sm2_qc','sh_sm8_%','sh_sm8_qc','sh_sm20_%','sh_sm20_qc','sh_st2_C','sh_st2_qc','sh_st8_C','sh_st2_qc','sh_st20_C','sh_st20_qc', 'depth_lower','depth_upper','depth_decrease','depth_increase','prec_lower','prec_upper','prec_decrease','prec_increase'], axis=1)
    #NRCS profile range check
qc_SWE['sh_SWE_qc'] = np.where((qc_SWE['sh_SWE_mm'] < qc_SWE['SWE_lower']), 'S', qc_SWE['sh_SWE_qc'])
qc_SWE['sh_SWE_qc'] = np.where((qc_SWE['sh_SWE_mm'] > qc_SWE['SWE_upper']), 'S', qc_SWE['sh_SWE_qc'])
    # Absolute value range check
qc_SWE['sh_SWE_auto_mm'] = np.where((qc_SWE['sh_SWE_mm'] < SWEmin), 0, qc_SWE['sh_SWE_mm'])
qc_SWE['sh_SWE_auto_mm'] = np.where((qc_SWE['sh_SWE_mm'] > qc_SWE['SWE_upper']), np.nan, qc_SWE['sh_SWE_auto_mm'])
    #Calculate median values for manual QC 
qc_SWE['sh_SWE_auto_mm'] = qc_SWE.sh_SWE_auto_mm.rolling(6, center=True, closed='right').median() 
qc_SWE['sh_SWE_auto24h_mm'] = qc_SWE.sh_SWE_auto_mm.rolling(24, center=True, closed='right').median() 
    #Set automated QC Flags
qc_SWE['sh_SWE_auto_qcflag'] = qc_SWE['sh_SWE_auto_mm'].isnull().map({True: 'S', False: 'E'})
qc_SWE['sh_SWE_auto_qcflag'] = np.where((qc_SWE['sh_SWE_mm'] == qc_SWE['sh_SWE_auto_mm']) , 'V', qc_SWE['sh_SWE_auto_qcflag'])

#### Snow Depth ####
qc_depth = copy.deepcopy(snotel_hourly)
qc_depth = qc_depth.drop(['sh_temp_C','sh_temp_qc','sh_precip_mm','sh_precip_qc','sh_SWE_mm','sh_SWE_qc','sh_sm2_%','sh_sm2_qc','sh_sm8_%','sh_sm8_qc','sh_sm20_%','sh_sm20_qc','sh_st2_C','sh_st2_qc','sh_st8_C','sh_st2_qc','sh_st20_C','sh_st20_qc','SWE_lower','SWE_upper','SWE_decrease','SWE_increase','prec_lower','prec_upper','prec_decrease','prec_increase'], axis=1)
    #NRCS profile range check
qc_depth['sh_snowdepth_qc'] = np.where((qc_depth['sh_snowdepth_cm'] < qc_depth['depth_lower']), 'S', qc_depth['sh_snowdepth_qc'])
qc_depth['sh_snowdepth_qc'] = np.where((qc_depth['sh_snowdepth_cm'] > qc_depth['depth_upper']), 'S', qc_depth['sh_snowdepth_qc'])
    # Autoclean range check
qc_depth['sh_depth_auto_cm'] = np.where((qc_depth['sh_snowdepth_cm'] < depthmin) , 0, qc_depth['sh_snowdepth_cm'])
qc_depth['sh_depth_auto_cm'] = np.where((qc_depth['sh_depth_auto_cm'] > qc_depth['depth_upper']) , np.nan, qc_depth['sh_depth_auto_cm'])
    #Fill missing data with linear interpolation
qc_depth['sh_depth_auto_cm'] = qc_depth['sh_depth_auto_cm'].interpolate(method ='linear', limit_direction ='backward') #should I add a limit?
    #Calculate median values for manual QC
qc_depth['sh_depth_auto12h_cm'] = qc_depth.sh_depth_auto_cm.rolling(12, center=True, closed='right').median()
qc_depth['sh_depth_auto24h_cm'] = qc_depth.sh_depth_auto_cm.rolling(24, center=True, closed='right').median()
    #Set automated QC Flags
qc_depth['sh_depth_auto_qcflag'] = qc_depth['sh_snowdepth_cm'].isnull().map({True: 'K', False: 'E'})
qc_depth['sh_depth_auto_qcflag'] = np.where((qc_depth['sh_snowdepth_cm'] == qc_depth['sh_depth_auto12h_cm']) , 'V', qc_depth['sh_depth_auto_qcflag'])

#### SNOTEL Precip ####
qc_precip = copy.deepcopy(snotel_hourly)
qc_precip = qc_precip.drop(['sh_temp_C','sh_temp_qc','sh_snowdepth_cm','sh_snowdepth_qc','sh_SWE_mm','sh_SWE_qc','sh_sm2_%','sh_sm2_qc','sh_sm8_%','sh_sm8_qc','sh_sm20_%','sh_sm20_qc','sh_st2_C','sh_st2_qc','sh_st8_C','sh_st2_qc','sh_st20_C','sh_st20_qc','depth_lower','depth_upper','depth_decrease','depth_increase','SWE_lower','SWE_upper','SWE_decrease','SWE_increase'], axis=1)
    #NRCS profile range check
qc_precip['sh_precip_qc'] = np.where((qc_precip['sh_precip_mm'] < qc_precip['prec_lower']), 'S', qc_precip['sh_precip_qc'])
qc_precip['sh_precip_qc'] = np.where((qc_precip['sh_precip_mm'] > qc_precip['prec_upper']), 'S', qc_precip['sh_precip_qc'])
    #Calculate median values for manual QC
qc_precip['sh_precip_auto_mm'] = qc_precip.sh_precip_mm.rolling(24, center=True, closed='right').median()
    #Set automated QC Flags
qc_precip['sh_precip_auto_qcflag'] = qc_precip['sh_precip_auto_mm'].isnull().map({True: 'S', False: 'E'})
qc_precip['sh_precip_auto_qcflag'] = np.where((qc_precip['sh_precip_auto_mm'].diff(24) > precip24hrmax) | (qc_precip['sh_precip_auto_mm'].diff()  > precip1hrmax), 'S', qc_precip['sh_precip_auto_qcflag'])
qc_precip['sh_precip_auto_qcflag'] = np.where((qc_precip['sh_precip_auto_mm'].diff() < 0), 'S', qc_precip['sh_precip_auto_qcflag'])
qc_precip['sh_precip_auto_qcflag'] = np.where((qc_precip['sh_precip_mm'] == qc_precip['sh_precip_auto_mm']) , 'V', qc_precip['sh_precip_auto_qcflag'])

#### WRCC Precip ####
wrcc_precip = copy.deepcopy(wrcc)
wrcc_precip = wrcc.drop(['w10_solarrad','w10_solarreflect','w10_unknown1','w10_unknown2','w10_unknown3','w10_mxtempF', 'w10_mntempF', 'w10_avtempF','w10_mxrh%', 'w10_mnrh%', 'w10_rh%', 'w10_bpinhg', 'w10_solarrad2','w10_snowdepthin', 'w10_mxsnowdepthin', 'w10_minsnowdepthin', 'w10_snowstd', 'w10_SWEin'], axis=1)
wrcc_precip['w10_precip_auto'] = np.where(wrcc_precip['w10_geonorstd']>5, np.NaN, wrcc_precip['w10_geonorcm'])
wrcc_precip['w10_precip_auto'] = wrcc_precip['w10_precip_auto'].diff()
wrcc_precip['w10_precip_auto'] = np.where(wrcc_precip['w10_precip_auto'] < -0.2, 0, wrcc_precip['w10_precip_auto'])
wrcc_precip['w10_precip_auto'] = wrcc_precip['w10_precip_auto'].cumsum()*10
wrcc_precip['w10_precip_auto'] = wrcc_precip['w10_precip_auto'].interpolate()
wrcc_precip = wrcc_precip.set_index('date') 
wrcc_precip = wrcc_precip.resample('h').max() #this takes the highest value which should be the value for the time stamp being selected
wrcc_precip.reset_index(inplace=True) 
    #Set automated QC Flags
wrcc_precip['w10_precip_auto_qcflag'] = wrcc_precip['w10_precip_auto'].isnull().map({True: 'S', False: 'V'})
wrcc_precip['w10_precip_auto_qcflag'] = np.where((wrcc_precip['w10_precip_auto'].diff() < 0), 'S', wrcc_precip['w10_precip_auto_qcflag'])

#### SNOTEL Temp ####
qc_temp = copy.deepcopy(snotel_hourly)
qc_temp = qc_temp.drop(['sh_precip_mm','sh_precip_qc','sh_snowdepth_cm','sh_snowdepth_qc','sh_SWE_mm','sh_SWE_qc','sh_sm2_%','sh_sm2_qc','sh_sm8_%','sh_sm8_qc','sh_sm20_%','sh_sm20_qc','sh_st2_C','sh_st2_qc','sh_st8_C','sh_st2_qc','sh_st20_C','sh_st20_qc'], axis=1)
#correct temp data - algorith from https://westernsnowconference.org/files/PDFs/2019Brown.pdf
qc_temp['sh_tempcorrected_C'] = 0.00000002 * qc_temp['sh_temp_C']**5 - 0.00000084*qc_temp['sh_temp_C']**4 - 0.00006726*qc_temp['sh_temp_C']**3 + 0.00246670*qc_temp['sh_temp_C']**2 + 1.07255015 * qc_temp['sh_temp_C'] - 1.16329887
#Set automated QC Flags
qc_temp['sh_temp_qcflag'] = qc_temp['sh_tempcorrected_C'].isnull().map({True: 'S', False: 'E'})

#### Soil Moisture ####
qc_sm = copy.deepcopy(snotel_hourly)
qc_sm = qc_sm.drop(['sh_temp_C','sh_temp_qc','sh_precip_mm','sh_precip_qc','sh_snowdepth_cm','sh_snowdepth_qc','sh_SWE_mm','sh_SWE_qc','sh_st2_C','sh_st2_qc','sh_st8_C','sh_st2_qc','sh_st20_C','sh_st20_qc','SWE_lower', 'SWE_upper', 'SWE_decrease', 'SWE_increase', 'prec_lower', 'prec_upper', 'prec_decrease', 'prec_increase'], axis=1)
qc_sm['sh_sm2_C_qaflag'] = 'A' #already QC'd by NRCS
qc_sm['sh_sm8_C_qaflag'] = 'A' #already QC'd by NRCS
qc_sm['sh_sm20_C_qaflag'] = 'A' #already QC'd by NRCS

#### Soil Temperature ####
qc_st = copy.deepcopy(snotel_hourly)
qc_st = qc_st.drop(['sh_temp_C','sh_temp_qc','sh_precip_mm','sh_precip_qc','sh_snowdepth_cm','sh_snowdepth_qc','sh_SWE_mm','sh_SWE_qc','sh_sm2_%','sh_sm2_qc','sh_sm8_%','sh_sm8_qc','sh_sm20_%','sh_sm20_qc','SWE_lower', 'SWE_upper', 'SWE_decrease', 'SWE_increase', 'prec_lower', 'prec_upper', 'prec_decrease', 'prec_increase'], axis=1)
qc_st['sh_st2_C_qaflag'] = 'A' #already QC'd by NRCS
qc_st['sh_st8_C_qaflag'] = 'A' #already QC'd by NRCS
qc_st['sh_st20_C_qaflag'] = 'A' #already QC'd by NRCS

#%%
#%% EXPORT DATA FOR MANUAL QC
### THIS SHOULD ONLY BE RUN ONCE
### COMMENT THIS CELL OUT AFTER THE INITIAL RUN
### RUNNING IT AFTER STARTING THE MANUAL QC PROCESS WILL OVERWRITE MANUAL EDITS

qc_SWE = qc_SWE.set_index('date')
qc_SWE['sh_SWE_1hrdiff'] = qc_SWE['sh_SWE_auto_mm'].diff() #to fill in manual difference in excell once SWE is verified
qc_SWE['sh_SWE_manual_mm'] = qc_SWE['sh_SWE_auto_mm']
qc_SWE['sh_SWE_manual_qcflag'] = qc_SWE['sh_SWE_auto_qcflag']
qc_SWE['sh_SWE_manual_qaflag'] = 'F' #Flag data QA as passing automated QC

qc_depth = qc_depth.set_index('date')
qc_depth['sh_depth_manual_cm'] = qc_depth['sh_depth_auto12h_cm']
qc_depth['sh_depth_manual_qcflag'] = qc_depth['sh_depth_auto_qcflag']
qc_depth['sh_depth_manual_qaflag'] = 'F' #Flag data QA as passing automated QC
qc_depth['density'] = qc_SWE['sh_SWE_manual_mm']/qc_depth['sh_depth_manual_cm']*10

qc_precip = qc_precip.set_index('date')
qc_SWE['sh_precip_1hrdiff'] = qc_precip['sh_precip_auto_mm'].diff() 
qc_precip['sh_precip_manual_mm'] = qc_precip['sh_precip_auto_mm']
qc_precip['sh_precip_manual_qcflag'] = qc_precip['sh_precip_auto_qcflag']
qc_precip['sh_precip_manual_qaflag'] = 'F' #Flag data QA as passing automated QC

wrcc_precip = wrcc_precip.set_index('date')
wrcc_precip = wrcc_precip.drop(['w10_geonorfeqhz','w10_geonorstd', 'w10_geonorcm'], axis=1)
wrcc_precip['w10_precip_1hrdiff'] = wrcc_precip['w10_precip_auto'].diff()
wrcc_precip['w10_precip_manual_mm'] = wrcc_precip['w10_precip_auto']
wrcc_precip['w10_precip_manual_qcflag'] = wrcc_precip['w10_precip_auto_qcflag']
wrcc_precip['w10_precip_manual_qaflag'] = 'F' #Flag data QA as passing automated QC

qc_temp = qc_temp.set_index('date')
qc_temp['sh_temp_qaflag'] = 'F' #Flag data QA as passing automated QC

qc_sm = qc_sm.set_index('date')

qc_st = qc_st.set_index('date')

#Create Preliminary data frame
preliminary = pd.concat([qc_SWE, qc_depth, qc_precip, wrcc_precip, qc_temp,qc_sm,qc_st], axis=1)
preliminary = preliminary.round(1)

#%% ###EXPORT PRELIMINARY FILE ONLY ONCE TO AVOID OVERWRITTING DATA ###
preliminary = preliminary.drop(['SWE_lower', 'SWE_upper', 'SWE_decrease', 'SWE_increase', 
                              'depth_lower', 'depth_upper', 'depth_decrease', 'depth_increase', 
                              'prec_lower', 'prec_upper', 'prec_decrease', 'prec_increase'], axis=1) #name columns

path = '/Users/anne/OneDrive/Data/QC_for_SRA/pubication_test/'+stationid+'_WY'+year+'_prelim.csv'
preliminary.to_csv(path)