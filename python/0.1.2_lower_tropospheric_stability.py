

# qsub -I -q normal -P nf33 -l walltime=3:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46+gdata/qx55


# region import packages


import xarray as xr
import glob
from netCDF4 import Dataset
from datetime import datetime, timedelta
from cdo import Cdo
cdo=Cdo()
import tempfile
import time
from metpy.calc import vertical_velocity_pressure, mixing_ratio_from_specific_humidity, potential_temperature
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
izlev = 6
izlev2 = 6
datasets = {}
for ids in dss: datasets[ids] = {}
LTS = {}
for ids in dss: LTS[ids] = {}

nrow = 1
ncol = len(dss)
fm_bottom = 1.5 / (4.4*nrow + 2)

datasets['UM'][f'z{izlev}1H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.z{izlev}.zarr').sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))
datasets['UM'][f'z{izlev}3H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT3H.z{izlev}.zarr').sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))

datasets['ICON'][f'z{izlev2}3Hm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/PT3H_mean_z{izlev2}_atm.zarr').sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))
datasets['ICON'][f'z{izlev2}1Dm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z{izlev2}_atm.zarr').sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))
datasets['ICON'][f'z{izlev2}1Dm']['pressure'] = datasets['ICON'][f'z{izlev2}1Dm']['pressure'] / 100

datasets['ERA5']['sl'] = {}
datasets['ERA5']['sl']['t2m'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2t/{year}/2t_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t2m'].rename({'latitude': 'lat', 'longitude': 'lon'})
datasets['ERA5']['sl']['sp'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/sp/{year}/sp_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['sp'].rename({'latitude': 'lat', 'longitude': 'lon'})
datasets['ERA5']['sl']['msl'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/msl/{year}/msl_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['msl'].rename({'latitude': 'lat', 'longitude': 'lon'})

datasets['ERA5']['pl'] = {}
datasets['ERA5']['pl']['t'] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/t/{year}/t_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t'].rename({'latitude': 'lat', 'longitude': 'lon'})


for iplev in ['psl']:
    # iplev='ps'
    print(f'#-------------------------------- {iplev}')
    
    if iplev == 'ps':
        LTS['ERA5'][iplev] = (potential_temperature(700 * units.hPa, datasets['ERA5']['pl']['t'].sel(level=700) * units.K) - potential_temperature(
            datasets['ERA5']['sl']['sp'] * units.Pa,
            datasets['ERA5']['sl']['t2m'] * units.K)).mean(dim='time').compute()
        LTS['UM'][iplev] = (potential_temperature(700 * units.hPa, datasets['UM'][f'z{izlev}3H']['ta'].sel(pressure=700) * units.K) - potential_temperature(
            datasets['UM'][f'z{izlev}1H']['ps'] * units.Pa,
            datasets['UM'][f'z{izlev}1H']['tas'] * units.K).resample(time='3h').mean()).mean(dim='time').compute()
        LTS['ICON'][iplev] = (potential_temperature(700 * units.hPa, datasets['ICON'][f'z{izlev2}1Dm']['ta'].sel(pressure=700) * units.K) - potential_temperature(
            datasets['ICON'][f'z{izlev2}3Hm']['ps'] * units.Pa,
            datasets['ICON'][f'z{izlev2}3Hm']['tas'] * units.K).resample(time='1d').mean()).mean(dim='time').compute()
    elif iplev == 'psl':
        LTS['ERA5'][iplev] = (potential_temperature(700 * units.hPa, datasets['ERA5']['pl']['t'].sel(level=700) * units.K) - potential_temperature(
            datasets['ERA5']['sl']['msl'] * units.Pa,
            datasets['ERA5']['sl']['t2m'] * units.K)).mean(dim='time').compute()
        LTS['UM'][iplev] = (potential_temperature(700 * units.hPa, datasets['UM'][f'z{izlev}3H']['ta'].sel(pressure=700) * units.K) - potential_temperature(
            datasets['UM'][f'z{izlev}1H']['psl'] * units.Pa,
            datasets['UM'][f'z{izlev}1H']['tas'] * units.K).resample(time='3h').mean()).mean(dim='time').compute()
        LTS['ICON'][iplev] = (potential_temperature(700 * units.hPa, datasets['ICON'][f'z{izlev2}1Dm']['ta'].sel(pressure=700) * units.K) - potential_temperature(
            datasets['ICON'][f'z{izlev2}3Hm']['psl'] * units.Pa,
            datasets['ICON'][f'z{izlev2}3Hm']['tas'] * units.K).resample(time='1d').mean()).mean(dim='time').compute()
    elif iplev == '1000hPa':
        LTS['ERA5'][iplev] = (potential_temperature(700 * units.hPa, datasets['ERA5']['pl']['t'].sel(level=700) * units.K) - potential_temperature(1000 * units.hPa, datasets['ERA5']['pl']['t'].sel(level=1000))).mean(dim='time').compute()
        LTS['UM'][iplev] = (potential_temperature(700 * units.hPa, datasets['UM'][f'z{izlev}3H']['ta'].sel(pressure=700) * units.K) - potential_temperature(1000 * units.hPa, datasets['UM'][f'z{izlev}3H']['ta'].sel(pressure=1000) * units.K)).mean(dim='time').compute()
        LTS['ICON'][iplev] = (potential_temperature(700 * units.hPa, datasets['ICON'][f'z{izlev2}1Dm']['ta'].sel(pressure=700) * units.K) - potential_temperature(1000 * units.hPa, datasets['ICON'][f'z{izlev2}1Dm']['ta'].sel(pressure=1000) * units.K)).mean(dim='time').compute()
    
    for imode in ['org', 'diff']:
        # imode = 'org'
        print(f'#---------------- {imode}')
        
        opng = f"figures/hackathon/0.1.0.2 LTS in {', '.join(dss)} {iplev} {imode} zlev{izlev}_{izlev2} {year}{month:02d}.png"
        
        fig, axs = plt.subplots(
            nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
            subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
            gridspec_kw={'hspace': 0.01, 'wspace': 0.01})
        
        if imode in ['org', 'regrid']:
            plt_colnames = dss
        elif imode in ['diff']:
            plt_colnames = [dss[0]] + [f'{ids} - {dss[0]}' for ids in dss[1:]]
        
        for jcol in range(ncol):
            axs[jcol] = globe_plot(ax_org=axs[jcol])
            axs[jcol].text(
                0,1.02,f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                ha='left', va='bottom', transform=axs[jcol].transAxes)
        
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=32, cm_interval1=2, cm_interval2=4, cmap='Oranges_r')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG_r')
        extend2 = 'both'
        
        plt_mesh = axs[0].pcolormesh(
            LTS['ERA5'][iplev].lon,
            LTS['ERA5'][iplev].lat,
            LTS['ERA5'][iplev],
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
        
        if imode == 'org':
            axs[1].set_global()
            axs[2].set_global()
            egh.healpix_show(
                    LTS[dss[1]][iplev], ax=axs[1], norm=pltnorm, cmap=pltcmp)
            egh.healpix_show(
                    LTS[dss[2]][iplev], ax=axs[2], norm=pltnorm, cmap=pltcmp)
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
                ax=axs, format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.3, fm_bottom-0.05, 0.4, 0.04]))
            cbar.ax.set_xlabel(era5_varlabels['LTS'])
        elif imode == 'diff':
            cells=get_nn_lon_lat_index(
                hp.get_nside(LTS['UM'][iplev]),#2**izlev,
                LTS['ERA5'][iplev].lon.values,
                LTS['ERA5'][iplev].lat.values)
            plt_mesh2 = axs[1].pcolormesh(
                LTS['ERA5'][iplev].lon, LTS['ERA5'][iplev].lat,
                LTS['UM'][iplev].isel(cell=cells) - LTS['ERA5'][iplev],
                norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())
            cells=get_nn_lon_lat_index(
                hp.get_nside(LTS['ICON'][iplev]),#2**izlev,
                LTS['ERA5'][iplev].lon.values,
                LTS['ERA5'][iplev].lat.values)
            axs[2].pcolormesh(
                LTS['ERA5'][iplev].lon, LTS['ERA5'][iplev].lat,
                LTS['ICON'][iplev].isel(cell=cells) - LTS['ERA5'][iplev],
                norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())
            
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
                ax=axs, format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.04]))
            cbar.ax.set_xlabel(era5_varlabels['LTS'])
            cbar2 = fig.colorbar(
                plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                ax=axs, format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks2, extend=extend2,
                cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.04]))
            cbar2.ax.set_xlabel(f'Difference in {era5_varlabels['LTS']}')
        
        fig.subplots_adjust(left=0.001, right=0.999, bottom=fm_bottom, top=0.94)
        fig.savefig(opng)







'''
datasets['ICON'][f'z{izlev2}1Dp'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_point_z{izlev2}_atm.zarr') # no tas/ps # w psl/ta
datasets['ICON'][f'z{izlev2}1Hp'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/PT1H_point_z{izlev2}_atm.zarr') # no tas/ps/ta # w psl
datasets['ICON'][f'z{izlev2}3Hp'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/PT3H_point_z{izlev2}_atm.zarr') # no tas/ps/ta # w psl
'''
# endregion





