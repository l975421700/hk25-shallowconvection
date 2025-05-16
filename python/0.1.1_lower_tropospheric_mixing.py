

# qsub -I -q normal -P nf33 -l walltime=1:00:00,ncpus=1,mem=60GB,jobfs=100MB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46+gdata/qx55


# region import packages


import xarray as xr
import glob
from netCDF4 import Dataset
from datetime import datetime, timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import string
from matplotlib import cm
import numpy as np
import os
import sys
sys.path.append(os.getcwd() + '/code/hk25-shallowconvection/module')
from plot_funcs import (globe_plot, remove_trailing_zero_pos, plt_mesh_pars)
from data_process_funcs import (read_MCD06COSP_M3, modis_cmip6_var)
from namelist import cmip6_era5_var, era5_varlabels
import easygems.healpix as egh
import healpy as hp
from matplotlib.colors import BoundaryNorm

# endregion


# region calculate LTM


dss = ['ERA5', 'UM', 'ICON']
izlev = 5
datasets = {}
for ids in dss: datasets[ids] = {}
LTM = {}
for ids in dss: LTM[ids] = {}

datasets['UM'][f'z{izlev}3H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT3H.z{izlev}.zarr')
datasets['ICON'][f'z{izlev}1Dm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z{izlev}_atm.zarr')

datasets['ERA5']['daily'] = {}
datasets['ERA5']['daily']['t'] = xr.open_mfdataset(sorted(glob.glob('/g/data/rt52/era5/pressure-levels/reanalysis/t/202[0-1]/*.nc')))
datasets['ERA5']['daily']['w'] = xr.open_mfdataset(sorted(glob.glob('/g/data/rt52/era5/pressure-levels/reanalysis/w/202[0-1]/*.nc')))
datasets['ERA5']['daily']['r'] = xr.open_mfdataset(sorted(glob.glob('/g/data/rt52/era5/pressure-levels/reanalysis/r/202[0-1]/*.nc')))

# lower tropospheric mixing
LTM['ERA5']['S']
LTM['UM']['S']
LTM['ICON']['S']



datasets['UM'][f'z{izlev}3H']['wa']
datasets['UM'][f'z{izlev}3H']['ta']
datasets['UM'][f'z{izlev}3H']['hur']

datasets['ICON'][f'z{izlev}1Dm']['wa']
datasets['ICON'][f'z{izlev}1Dm']['ta']
datasets['ICON'][f'z{izlev}1Dm']['hur']






'''
datasets['UM'][f'z{izlev}3H'].data_vars
datasets['ICON'][f'z{izlev}1Dm'].data_vars

datasets['ERA5']['mon'] = {}
datasets['ERA5']['mon']['t'] = xr.open_mfdataset(sorted(glob.glob('/g/data/rt52/era5/pressure-levels/monthly-averaged/t/202[0-1]/*.nc')))
datasets['ERA5']['mon']['w'] = xr.open_mfdataset(sorted(glob.glob('/g/data/rt52/era5/pressure-levels/monthly-averaged/w/202[0-1]/*.nc')))
datasets['ERA5']['mon']['r'] = xr.open_mfdataset(sorted(glob.glob('/g/data/rt52/era5/pressure-levels/monthly-averaged/r/202[0-1]/*.nc')))

'''
# endregion

