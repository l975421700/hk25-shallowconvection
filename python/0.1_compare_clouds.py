

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


# endregion


# region import data



'''
fl = sorted(glob.glob('data/obs/MODIS/MCD06COSP_M3/*/*/*.nc'))
ivar2 = 'Cloud_Mask_Fraction_Low'
ivar1 = modis_cmip6_var[ivar2]

xr.concat([read_MCD06COSP_M3(ifile, ivar2, ivar1) for ifile in fl], dim='time')



import xarray as xr
import easygems.healpix as egh

ds = xr.open_zarr('/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.z10.zarr')
ds.tas


# MODIS Calibrated Radiances: MOD021KM, MYD021KM
# MODIS Cloud Mask: MOD35_L2, MYD35_L2
# MCD06COSP_M3_MODIS - MODIS (Aqua/Terra) Cloud Properties Level 3 monthly, 1x1 degree grid

print(ds.groups.keys())
print(ds.variables.keys())

'''
# endregion

