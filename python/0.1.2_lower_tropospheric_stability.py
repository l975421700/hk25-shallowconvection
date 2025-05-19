

# region import packages


import xarray as xr
import glob
from netCDF4 import Dataset
from datetime import datetime, timedelta
from cdo import Cdo
cdo=Cdo()
import tempfile
import time
from metpy.calc import vertical_velocity_pressure, mixing_ratio_from_specific_humidity
from metpy.units import units
import intake
import pandas as pd

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
from plot_funcs import (globe_plot, remove_trailing_zero_pos, plt_mesh_pars, get_nn_lon_lat_index)
from data_process_funcs import (read_MCD06COSP_M3, modis_cmip6_var)
from namelist import cmip6_era5_var, era5_varlabels
import easygems.healpix as egh
import healpy as hp
from matplotlib.colors import BoundaryNorm
import calendar


# endregion


# region plot LTS

year = 2020
month = 6

dss = ['ERA5', 'UM', 'ICON']
izlev = 5
izlev2 = 5
datasets = {}
for ids in dss: datasets[ids] = {}
LTS = {}
for ids in dss: LTS[ids] = {}

datasets['UM'][f'z{izlev}1H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.z{izlev}.zarr').sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))
datasets['UM'][f'z{izlev}3H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT3H.z{izlev}.zarr').sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))

datasets['ICON'][f'z{izlev2}1Dm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z{izlev2}_atm.zarr').sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))

datasets['ERA5']['sl'] = {}
datasets['ERA5']['sl']['2t'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2t/{year}/2t_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t2m'].rename({'latitude': 'lat', 'longitude': 'lon'})
datasets['ERA5']['sl']['sp'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/sp/{year}/sp_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['sp'].rename({'latitude': 'lat', 'longitude': 'lon'})

datasets['ERA5']['pl'] = {}
datasets['ERA5']['pl']['t'] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/t/{year}/t_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t']


datasets['UM'][f'z{izlev}1H']['tas']
datasets['UM'][f'z{izlev}1H']['ps']
datasets['UM'][f'z{izlev}3H']['ta']

datasets['ICON'][f'z{izlev2}1Dm']['tas']
datasets['ICON'][f'z{izlev2}1Dm']['ps']
datasets['ICON'][f'z{izlev2}1Dm']['ta']

datasets['ERA5']['sl']['2t']
datasets['ERA5']['sl']['sp']
datasets['ERA5']['pl']['t']


# endregion





