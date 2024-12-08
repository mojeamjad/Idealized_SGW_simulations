# -*- coding: utf-8 -*-
# %%
"""
Created on Fri Nov 22 09:13:54 2024

@author: Mozhgan
"""

import netCDF4 as nc
import pandas as pd
import joblib as jb
import numpy as np
# %%
file_list = pd.read_csv('file_list.csv', header=0)
print(file_list)
# %% Figure Sup
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
# %%

all_data ={}
for i in range(len(file_list)):
    print(file_list['Test_name'][i])
    ds = nc.Dataset(file_list['path'][i]+file_list['filename'][i])
    #########################
    lat = ds['lat'][:]
    lon = ds['lon'][:]
    lat0 = int((lat[:].shape)[0]/2)+1
    lon0 = int((lon[:].shape)[0]/2)+1
    z_mc_r = ds['z_mc'][:] 
    z_mc = (z_mc_r[:,20,20]).compressed()# m to km
    
    
    test_info = {}
    u = ds['u'][-1,...]
    m_u = u.mean(axis=-1).mean(axis=-1)
    u_p = u - m_u[:, np.newaxis, np.newaxis]
    w = ds['u'][-1,...]
    m_w = w.mean(axis=-1).mean(axis=-1)
    w_p = w - m_w[:, np.newaxis, np.newaxis]
    rho = ds['rho'][-1,...]
    
    F_x = rho * w_p * u_p
    a_x = -np.gradient(F_x,z_mc,axis=0)/rho
    
    test_info['a_x(t,z,lat,lon)'] = a_x[:,lat0,lon0]
    all_data [file_list['Test_name'][i]] = test_info
jb.dump(all_data,'./selected_simulations_drag.pkl')  

# %%
plt.plot(a_x[:,lat0,lon0])

