# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:13:54 2024

@author: Mozhgan
"""

import netCDF4 as nc
import pandas as pd
import joblib as jb
import numpy as np
#%%
file_list = pd.read_csv('file_list.csv', header=0)
print(file_list)
#%% Figure Sup
all_data ={}
for i in range(len(file_list)):
    print(file_list['Test_name'][i])
    ds = nc.Dataset(file_list['path'][i]+file_list['filename'][i])
    ######## atributes
    lat = ds['lat'][:]
    lon = ds['lon'][:]
    lat0 = int((lat[:].shape)[0]/2)
    z_mc_r = ds['z_mc'][:]
    z_mc = z_mc_r[:,20,:]/1000# m to km
    time = round(ds['time'][-1]/60/24)
    print('selected time = day ',time)
    ######## U-wind height-longitude cross-section
    u = ds['u'][-1,...]
    ######## data dictionary
    test_info = {}
    test_info['lat'] = lat
    test_info['lon'] = lon
    test_info['time'] = time
    test_info['height'] = z_mc
    test_info['u(t,z,0,lon)'] = u[:,lat0,:]
    test_info['m_u(t,z)'] = u.mean(axis=-1).mean(axis=-1)
    test_info['std_up(z)'] = np.std(u, axis=(1, 2))
    #########
    all_data [file_list['Test_name'][i]] = test_info
# jb.dump(all_data,'./selected_simulations_time_short.pkl')    
#%% supplementary: special cases
file_list = pd.DataFrame()
file_list['Test_name'] = ['B3high','B3high_cx10','B3high_cz25','B3high_smag015','B3high_efdt36','B3asc','B3rsg']
file_list['path'] = ['limited_bell_xy_Fr_high_B3_1p2_5N5S',
                     'limited_bell_xy_Fr_high_B3_0p6_1p8S',
                     'limited_bell_xy_Fr_high_B3_1p2_5N5S_strong-SGW2',
                     'limited_bell_xy_Fr_high_B3_1p2_5p5NS_efdt10_smag015',
                     'limited_bell_xy_Fr_high_B3_1p2_5p5NS_efdt36_smag15',
                     'limited_bell_xy_B3asc',
                     'limited_bell_xy_B3rsg']
print(file_list)
#%%
all_data ={}
for i in range(len(file_list)):
    print(file_list['Test_name'][i])
    ds = nc.Dataset(file_list['path'][i]+'/bell_DOM01_ML_0009.nc')
    ######## atributes
    lat = ds['lat'][:]
    lon = ds['lon'][:]
    lat0 = int((lat[:].shape)[0]/2)
    z_mc_r = ds['z_mc'][:]
    z_mc = z_mc_r[:,20,:]/1000# m to km
    time = round(ds['time'][-1]/60/24)
    print('selected time = day ',time)
    ######## U-wind height-longitude cross-section
    u = ds['u'][0,...]
    ######## data dictionary
    test_info = {}
    test_info['lat'] = lat
    test_info['lon'] = lon
    test_info['time'] = time
    test_info['height'] = z_mc
    test_info['u(t,z,0,lon)'] = u[:,lat0,:]
    test_info['u(t,10,lat,lon)'] = u[len(z_mc)-20,:,:]
    test_info['u(t,100,lat,lon)'] = u[len(z_mc)-200,:,:]
    test_info['m_u(t,z)'] = u.mean(axis=-1).mean(axis=-1)
    test_info['std_up(z)'] = np.std(u, axis=(1, 2))

    #########
    all_data [file_list['Test_name'][i]] = test_info
jb.dump(all_data,'./special_cases_B3high.pkl') 
#%% supplementary: special cases
file_list = pd.DataFrame()
file_list['Test_name'] = ['B4mid','B4mid_cx10','B4mid_cz25','B4mid_smag015','B4mid_efdt36','B4mid_ey180']
file_list['path'] = ['limited_bell_xy_Fr_middle_B4_2p5_7N7S',
                     'limited_bell_xy_Fr_mid_B4_1p2_7p2EW',
                     'limited_bell_xy_Fr_mid_B4_0p6_3p2EW',
                     'limited_bell_xy_Fr_mid_B4_2p5_7N7S_efdt10_smag015',
                     'limited_bell_xy_Fr_mid_B4_2p5_7N7S_efdt36_smag15',
                     'limited_bell_xy_Fr_mid_B4_2p5_11p5NS']
print(file_list)
#%%
all_data ={}
for i in range(len(file_list)):
    print(file_list['Test_name'][i])
    ds = nc.Dataset(file_list['path'][i]+'/bell_DOM01_ML_0049.nc')
    ######## atributes
    lat = ds['lat'][:]
    lon = ds['lon'][:]
    lat0 = int((lat[:].shape)[0]/2)
    z_mc_r = ds['z_mc'][:]
    z_mc = z_mc_r[:,20,:]/1000# m to km
    time = round(ds['time'][-1]/60/24)
    print('selected time = day ',time)
    ######## U-wind height-longitude cross-section
    u = ds['u'][0,...]
    ######## data dictionary
    test_info = {}
    test_info['lat'] = lat
    test_info['lon'] = lon
    test_info['time'] = time
    test_info['height'] = z_mc
    test_info['u(t,z,0,lon)'] = u[:,lat0,:]
    test_info['u(t,10,lat,lon)'] = u[np.argmin(np.abs(z_mc[:,0] - 10)),:,:]
    test_info['u(t,100,lat,lon)'] = u[np.argmin(np.abs(z_mc[:,0] - 100)),:,:]
    test_info['m_u(t,z)'] = u.mean(axis=-1).mean(axis=-1)
    test_info['std_up(z)'] = np.std(u, axis=(1, 2))

    #########
    all_data [file_list['Test_name'][i]] = test_info
jb.dump(all_data,'./special_cases_B4mid.pkl') 
#%% supplementary: special cases
file_list = pd.DataFrame()
file_list['Test_name'] = ['B5low','B5low_smag015','B5low_efdt36']
file_list['path'] = ['limited_bell_xy_Fr_low_B5_5_15NS',
                     'limited_bell_xy_Fr_low_B5_5_15NS_efdt10_smag015',
                     'limited_bell_xy_Fr_low_B5_5_15NS_efdt36_smag15']
print(file_list)
#%%
all_data ={}
for i in range(len(file_list)):
    print(file_list['Test_name'][i])
    ds = nc.Dataset(file_list['path'][i]+'/bell_DOM01_ML_0049.nc')
    ######## atributes
    lat = ds['lat'][:]
    lon = ds['lon'][:]
    lat0 = int((lat[:].shape)[0]/2)
    z_mc_r = ds['z_mc'][:]
    z_mc = z_mc_r[:,20,:]/1000# m to km
    time = round(ds['time'][-1]/60/24)
    print('selected time = day ',time)
    ######## U-wind height-longitude cross-section
    u = ds['u'][0,...]
    ######## data dictionary
    test_info = {}
    test_info['lat'] = lat
    test_info['lon'] = lon
    test_info['time'] = time
    test_info['height'] = z_mc
    test_info['u(t,z,0,lon)'] = u[:,lat0,:]
    test_info['u(t,10,lat,lon)'] = u[np.argmin(np.abs(z_mc[:,0] - 10)),:,:]
    test_info['u(t,100,lat,lon)'] = u[np.argmin(np.abs(z_mc[:,0] - 100)),:,:]
    test_info['m_u(t,z)'] = u.mean(axis=-1).mean(axis=-1)
    test_info['std_up(z)'] = np.std(u, axis=(1, 2))

    #########
    all_data [file_list['Test_name'][i]] = test_info
jb.dump(all_data,'./special_cases_B5low.pkl') 
#%%

import numpy as np
from matplotlib.ticker import (MultipleLocator,  AutoMinorLocator, ScalarFormatter)
import matplotlib as mlt
import matplotlib.pyplot as plt


ds = nc.Dataset('limited_bell_xy_Fr_high_B3_1p2_5N5S/bell_DOM01_ML_0009.nc')
######## atributes
lat = ds['lat'][:]
lon = ds['lon'][:]
lat0 = int((lat[:].shape)[0]/2)
z_mc_r = ds['z_mc'][:]
z_mc = z_mc_r[:,20,:]/1000# m to km
time = round(ds['time'][-1]/60/24)
print('selected time = day ',time)
######## U-wind height-longitude cross-section
u = ds['u'][0,...]
uh = u[len(z_mc)-200,:,:]
distancex = lon*110 # zonal distance (km)
distancey = lat*110 # meridional distance (km)
xv, yv = np.meshgrid(distancex, distancey)

plt.contour(xv,yv,uh,origin='upper')
