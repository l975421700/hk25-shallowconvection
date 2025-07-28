import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import string
import xarray as xr
# import easygems.healpix as egh
# import healpy as hp
from pathlib import Path
import os
import sys
sys.path.append('/home/548/cd3022/hk25-shallowconvection/module')
from plot_funcs import (globe_plot, remove_trailing_zero_pos, plt_mesh_pars)


# qsub -I -q normal -P nf33 -l walltime=2:00:00,ncpus=1,mem=120GB,jobfs=100MB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46+gdata/qx55+gdata/hh5
# source /scratch/nf33/public/hackathon_env/bin/activate

um = xr.open_zarr('/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.z5.zarr')
um_monthly = um.resample(time='1M').mean()
um_rsut = um_monthly['rsut'].mean(dim='cell').sel(time=slice('2020-03', '2021-02')) * -1
um_rsutcs = um_monthly['rsutcs'].mean(dim='cell').sel(time=slice('2020-03', '2021-02')) * -1
um_cloud_fraction = um_monthly['clt'].mean(dim='cell').sel(time=slice('2020-03', '2021-02'))

 
##### CERES Data
ceres = xr.open_dataset('/g/data/er8/users/cd3022/hk25-ShallowConvection/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202502.nc')

ceres['toa_sw_all_mon'] = ceres['toa_sw_all_mon'] * -1
ceres['toa_lw_all_mon'] = ceres['toa_lw_all_mon'] * -1
ceres = ceres.rename({'toa_sw_all_mon': 'rsut'})
ceres = ceres.rename({'toa_lw_all_mon': 'rlut'})

ceres_monthly = ceres.resample(time='1M').mean()
ceres_rsut = ceres_monthly['rsut'].mean(dim=['lat', 'lon']).sel(time=slice('2020-03', '2021-02'))
ceres_rsutcs = ceres_monthly['toa_sw_clr_c_mon'].mean(dim=['lat', 'lon']).sel(time=slice('2020-03', '2021-02')) * -1
ceres_cloud_fraction =  ceres_monthly['cldarea_total_daynight_mon'].mean(dim=['lat', 'lon']).sel(time=slice('2020-03', '2021-02')) / 100




# Plotting
fig, ax = plt.subplots()

ax.plot(ceres_rsut.time, ceres_rsut, label='CERES', c='C0', ls='-')
ax.plot(um_rsut.time, um_rsut, label='UM', c='C1', ls='-')

ax.plot(ceres_rsutcs.time, ceres_rsutcs, label='CERES: clear sky', c='C0', ls='--')
ax.plot(um_rsutcs.time, um_rsutcs, label='UM: clear sky', c='C1', ls='--')

ax.set_ylabel('TOA shortwave flux (W/m2)')
plt.legend()
plt.savefig('/home/548/cd3022/figures/HK25/monthly_sw.png')
plt.show()



fig, ax = plt.subplots()

ax.plot(ceres_cloud_fraction.time, ceres_cloud_fraction, label='CERES', c='C0', ls='-')
ax.plot(um_cloud_fraction.time, um_cloud_fraction, label='UM', c='C1', ls='-')

ax.set_ylabel('Cloud Area Fraction [-]')
plt.legend()
plt.savefig('/home/548/cd3022/figures/HK25/monthly_cloud_fraction.png')
plt.show()

# UM variables:
# clt: cloud fraction
# rsutcs: toa outgoing sw assuming clear sky

# CERES variables
# toa_sw_clr_c_mon: toa outgoing sw clear sky
# cldarea_total_daynight_mon: cloud area fraction