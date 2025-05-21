

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


# region calculate LTM-S


time1 = time.perf_counter()


dss = ['ERA5', 'UM', 'ICON']
izlev = 5
datasets = {}
for ids in dss: datasets[ids] = {}
LTM = {}
for ids in dss: LTM[ids] = {}

datasets['UM'][f'z{izlev}3H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT3H.z{izlev}.zarr').sel(time=slice('2020-03', '2021-02'))
datasets['ICON'][f'z{izlev}1Dm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z{izlev}_atm.zarr').sel(time=slice('2020-03', '2021-02'))

datasets['ERA5']['hourly'] = {}
for var1 in ['t', 'w', 'r']:
    datasets['ERA5']['hourly'][var1] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/pressure-levels/reanalysis/{var1}/202[0-1]/*.nc'))[2:14], combine='by_coords', parallel=True)[var1].rename({'latitude': 'lat', 'longitude': 'lon'})


LTM['ERA5']['S'] = (((datasets['ERA5']['hourly']['r'].sel(level=700).astype('float32') - datasets['ERA5']['hourly']['r'].sel(level=850).astype('float32')) / 100 - (datasets['ERA5']['hourly']['t'].sel(level=700).astype('float32') - datasets['ERA5']['hourly']['t'].sel(level=850).astype('float32')) / 9) / 2).mean(dim='time').compute()

LTM['UM']['S'] = (((datasets['UM'][f'z{izlev}3H']['hur'].sel(pressure=700) - datasets['UM'][f'z{izlev}3H']['hur'].sel(pressure=850)) / 100 - (datasets['UM'][f'z{izlev}3H']['ta'].sel(pressure=700) - datasets['UM'][f'z{izlev}3H']['ta'].sel(pressure=850)) / 9) / 2).mean(dim='time').compute()

LTM['ICON']['S'] = (((datasets['ICON'][f'z{izlev}1Dm']['hur'].sel(pressure=700 * 100) - datasets['ICON'][f'z{izlev}1Dm']['hur'].sel(pressure=850 * 100)) / 100 - (datasets['ICON'][f'z{izlev}1Dm']['ta'].sel(pressure=700 * 100) - datasets['ICON'][f'z{izlev}1Dm']['ta'].sel(pressure=850 * 100)) / 9) / 2).mean(dim='time').compute()


time2 = time.perf_counter()
print(f'Execution time: {time2 - time1:.1f} s')








'''
datasets['ICON'][f'z{izlev}1Dm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z{izlev}_atm.zarr')
datasets['ICON'][f'z{izlev}3Hp'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/PT3H_point_z{izlev}_atm.zarr')
datasets['ICON'][f'z{izlev}3Hm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/PT3H_mean_z{izlev}_atm.zarr')
datasets['UM'][f'z{izlev}3H'].data_vars
datasets['ICON'][f'z{izlev}1Dm'].data_vars

datasets['ERA5']['mon'] = {}
datasets['ERA5']['mon']['t'] = xr.open_mfdataset(sorted(glob.glob('/g/data/rt52/era5/pressure-levels/monthly-averaged/t/202[0-1]/*.nc')))
datasets['ERA5']['mon']['w'] = xr.open_mfdataset(sorted(glob.glob('/g/data/rt52/era5/pressure-levels/monthly-averaged/w/202[0-1]/*.nc')))
datasets['ERA5']['mon']['r'] = xr.open_mfdataset(sorted(glob.glob('/g/data/rt52/era5/pressure-levels/monthly-averaged/r/202[0-1]/*.nc')))

with tempfile.NamedTemporaryFile(suffix='.nc') as temp_output:
        cdo.mergetime(input=sorted(glob.glob('/g/data/rt52/era5/pressure-levels/reanalysis/t/202[0-1]/*.nc'))[2:14], output=temp_output.name)
        datasets['ERA5']['daily']['t'] = xr.open_dataset(temp_output.name)['t'].rename({'latitude': 'lat', 'longitude': 'lon'})

'''
# endregion


# region calculate LTM-D

dss = ['ERA5', 'UM', 'ICON']
izlev = 5
datasets = {}
for ids in dss: datasets[ids] = {}
LTM = {}
for ids in dss: LTM[ids] = {}
omega = {}
for ids in dss: omega[ids] = {}

datasets['UM'][f'z{izlev}3H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT3H.z{izlev}.zarr').sel(time=slice('2020-03', '2021-02'))
datasets['ICON'][f'z{izlev}1Dm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z{izlev}_atm.zarr').sel(time=slice('2020-03', '2021-02'))

datasets['ERA5']['hourly'] = {}
for var1 in ['w']:
    datasets['ERA5']['hourly'][var1] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/pressure-levels/reanalysis/{var1}/202[0-1]/*.nc'))[2:14], combine='by_coords', parallel=True)[var1].rename({'latitude': 'lat', 'longitude': 'lon'})

def get_omega(dataset, pressure):
    from metpy.calc import vertical_velocity_pressure, mixing_ratio_from_specific_humidity
    from metpy.units import units
    
    mixing_ratio = mixing_ratio_from_specific_humidity(
        dataset['hus'].sel(pressure=pressure) * units('kg/kg'))
    omega = vertical_velocity_pressure(
        dataset['wa'].sel(pressure=pressure) * units('m/s'),
        700 * units.hPa,
        dataset['ta'].sel(pressure=pressure) * units.K,
        mixing_ratio)
    return(omega)



for pressure in [400, 500, 600, 700, 850]:
    print(f'#-------- {pressure}')
    omega['ERA5'][f'{pressure}'] = datasets['ERA5']['hourly'][var1].sel(level=pressure).astype('float32')
    omega['UM'][f'{pressure}'] = get_omega(
        datasets['UM'][f'z{izlev}3H'], pressure).compute()
    omega['ICON'][f'{pressure}'] = get_omega(
        datasets['ICON'][f'z{izlev}1Dm'], pressure*100).compute()


def get_LTM_D(omega_in):
    # omega_in = omega['UM'] # omega['ERA5'] # omega['ICON'] #
    omega1 = (omega_in['850'] + omega_in['700']) / 2
    omega2 = (omega_in['600'] + omega_in['500'] + omega_in['400']) / 3
    LTM_D = (((omega2 - omega1).clip(min=0) * xr.where(-omega1 >= 0, 1, 0)) / (-omega2).clip(min=0)).mean(dim='time', skipna=True).compute()
    return(LTM_D)

get_LTM_D(omega['ERA5'])
get_LTM_D(omega['UM'])
get_LTM_D(omega['ICON'])







'''
datasets['ERA5']['hourly']['w']
['hus']

# check
ds1 = get_omega(datasets['ICON'][f'z{izlev}1Dm'], 700*100).compute()
ds2 = vertical_velocity_pressure(
    datasets['ICON'][f'z{izlev}1Dm']['wa'].sel(pressure=700*100) * units('m/s'),
    700 * units.hPa,
    datasets['ICON'][f'z{izlev}1Dm']['ta'].sel(pressure=700*100) * units.K,
    mixing_ratio_from_specific_humidity(
        datasets['ICON'][f'z{izlev}1Dm']['hus'].sel(pressure=700*100) * units('kg/kg'))).compute()
print((ds1 == ds2).all().values)

ds3 = get_omega(datasets['UM'][f'z{izlev}3H'], 700).compute()
ds4 = vertical_velocity_pressure(
    datasets['UM'][f'z{izlev}3H']['wa'].sel(pressure=700) * units('m/s'),
    700 * units.hPa,
    datasets['UM'][f'z{izlev}3H']['ta'].sel(pressure=700) * units.K,
    mixing_ratio_from_specific_humidity(
        datasets['UM'][f'z{izlev}3H']['hus'].sel(pressure=700) * units('kg/kg'))).compute()
print((ds3.values[np.isfinite(ds3.values)] == ds4.values[np.isfinite(ds4.values)]).all())


omega1 = np.mean([omega['UM'][pressure] for pressure in ['850', '700']], axis=0)
omega2 = np.mean([omega['UM'][pressure] for pressure in ['600', '500', '400']], axis=0)

ds1 = np.mean([omega['UM'][pressure] for pressure in ['850', '700']], axis=0)
ds2 = ((omega['UM']['850'] + omega['UM']['700']).values / 2)
print((ds1[np.isfinite(ds1)] == ds2[np.isfinite(ds2)]).all())
ds1 = np.mean([omega['UM'][pressure] for pressure in ['600', '500', '400']], axis=0)
ds2 = (omega['UM']['600'] + omega['UM']['500'] + omega['UM']['400']).values / 3
print((ds1[np.isfinite(ds1)] == ds2[np.isfinite(ds2)]).all())

'''
# endregion


# region plot one month of LTD S and D

year = 2020
month = 6

dss = ['ERA5', 'UM', 'ICON']
izlev = 10
izlev2 = 11
datasets = {}
for ids in dss: datasets[ids] = {}
LTM = {}
for ids in dss: LTM[ids] = {}
omega = {}
for ids in dss: omega[ids] = {}

datasets['UM'][f'z{izlev}3H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT3H.z{izlev}.zarr').sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))
datasets['ICON'][f'z{izlev2}1Dm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z{izlev2}_atm.zarr').sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))
datasets['ICON'][f'z{izlev2}1Dp'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_point_z{izlev2}_atm.zarr').sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))

datasets['ERA5']['hourly'] = {}
for var1 in ['t', 'w', 'r']:
    datasets['ERA5']['hourly'][var1] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].rename({'latitude': 'lat', 'longitude': 'lon'})


LTM['ERA5']['S'] = (((datasets['ERA5']['hourly']['r'].sel(level=700).astype('float32') - datasets['ERA5']['hourly']['r'].sel(level=850).astype('float32')) / 100 - (datasets['ERA5']['hourly']['t'].sel(level=700).astype('float32') - datasets['ERA5']['hourly']['t'].sel(level=850).astype('float32')) / 9) / 2).mean(dim='time').compute()

LTM['UM']['S'] = (((datasets['UM'][f'z{izlev}3H']['hur'].sel(pressure=700) - datasets['UM'][f'z{izlev}3H']['hur'].sel(pressure=850)) / 100 - (datasets['UM'][f'z{izlev}3H']['ta'].sel(pressure=700) - datasets['UM'][f'z{izlev}3H']['ta'].sel(pressure=850)) / 9) / 2).mean(dim='time').compute()

LTM['ICON']['S'] = (((datasets['ICON'][f'z{izlev2}1Dm']['hur'].sel(pressure=700 * 100) - datasets['ICON'][f'z{izlev2}1Dm']['hur'].sel(pressure=850 * 100)) / 100 - (datasets['ICON'][f'z{izlev2}1Dm']['ta'].sel(pressure=700 * 100) - datasets['ICON'][f'z{izlev2}1Dm']['ta'].sel(pressure=850 * 100)) / 9) / 2).mean(dim='time').compute()


def get_omega(dataset, pressure, pressure_unit = units.hPa):
    from metpy.calc import vertical_velocity_pressure, mixing_ratio_from_specific_humidity
    from metpy.units import units
    
    mixing_ratio = mixing_ratio_from_specific_humidity(
        dataset['hus'].sel(pressure=pressure) * units('kg/kg')).compute()
    omega = vertical_velocity_pressure(
        dataset['wa'].sel(pressure=pressure).compute() * units('m/s'),
        pressure * pressure_unit,
        dataset['ta'].sel(pressure=pressure).compute() * units.K,
        mixing_ratio).compute()
    return(omega)

for pressure in [400, 500, 600, 700, 850]:
    # pressure = 400
    print(f'#-------- {pressure}')
    omega['ERA5'][f'{pressure}'] = datasets['ERA5']['hourly']['w'].sel(level=pressure).astype('float32').compute()
    omega['UM'][f'{pressure}'] = get_omega(
        datasets['UM'][f'z{izlev}3H'], pressure)
    omega['ICON'][f'{pressure}'] = get_omega(
        # datasets['ICON'][f'z{izlev2}1Dm'],
        datasets['ICON'][f'z{izlev2}1Dp'],
        pressure*100, pressure_unit=units.Pa)

def get_LTM_D(omega_in):
    # omega_in = omega['UM'] # omega['ERA5'] # omega['ICON'] #
    omega1 = (omega_in['850'] + omega_in['700']) / 2
    omega2 = (omega_in['600'] + omega_in['500'] + omega_in['400']) / 3
    Ddata = (omega2 - omega1).clip(min=0) * xr.where(-omega1 >= 0, 1, 0) / (-omega2).clip(min=0)
    LTM_D = xr.where(np.isinf(Ddata), np.nan, Ddata).mean(dim='time', skipna=True).compute()
    return(LTM_D)

LTM['ERA5']['D'] = get_LTM_D(omega['ERA5'])
LTM['UM']['D'] = get_LTM_D(omega['UM'])
LTM['ICON']['D'] = get_LTM_D(omega['ICON'])


nrow = 1
ncol = len(dss)
fm_bottom = 1.5 / (4.4*nrow + 2)

for imode in ['org', 'diff']:
    # imode = 'org'
    # ['org', 'diff']
    print(f'#-------------------------------- {imode}')
    
    for var in ['S+D']:
        # var = 'S+D'
        # ['S', 'D', 'S+D']
        print(f'#---------------- {var}')
        
        opng = f"figures/hackathon/0.1.0.1 {var} in {', '.join(dss)} {imode} zlev{izlev}_{izlev2} {year}{month:02d}.png"
        
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
        
        if var in ['S']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='PuOr_r')
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='BrBG_r')
            extend2 = 'both'
        elif var in ['D', 'S+D']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=4, cm_interval1=0.25, cm_interval2=0.5, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-4, cm_max=4, cm_interval1=0.5, cm_interval2=1, cmap='BrBG_r')
            extend2 = 'both'
        
        if var in ['S+D']:
            plt_mesh = axs[0].pcolormesh(
                LTM['ERA5']['S'].lon,
                LTM['ERA5']['S'].lat,
                LTM['ERA5']['S'] + LTM['ERA5']['D'],
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
        elif var in ['S', 'D']:
            plt_mesh = axs[0].pcolormesh(
                LTM['ERA5']['S'].lon,
                LTM['ERA5']['S'].lat,
                LTM['ERA5'][var],
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
        
        if imode == 'org':
            axs[1].set_global()
            axs[2].set_global()
            if var in ['S+D']:
                egh.healpix_show(
                    LTM[dss[1]]['S'] + LTM[dss[1]]['D'],
                    ax=axs[1], norm=pltnorm, cmap=pltcmp)
                egh.healpix_show(
                    LTM[dss[2]]['S'] + LTM[dss[2]]['D'],
                    ax=axs[2], norm=pltnorm, cmap=pltcmp)
            elif var in ['S', 'D']:
                egh.healpix_show(
                    LTM[dss[1]][var],
                    ax=axs[1], norm=pltnorm, cmap=pltcmp)
                egh.healpix_show(
                    LTM[dss[2]][var],
                    ax=axs[2], norm=pltnorm, cmap=pltcmp)
            
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
                ax=axs, format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.3, fm_bottom-0.05, 0.4, 0.04]))
            cbar.ax.set_xlabel(era5_varlabels[var])
        elif imode == 'diff':
            cells=get_nn_lon_lat_index(
                hp.get_nside(LTM['UM']['S']),#2**izlev,
                LTM['ERA5']['S'].lon.values,
                LTM['ERA5']['S'].lat.values)
            if var in ['S+D']:
                plt_data = (LTM['UM']['S'] + LTM['UM']['D']).isel(cell=cells) - (LTM['ERA5']['S'] + LTM['ERA5']['D'])
            elif var in ['S', 'D']:
                plt_data = LTM['UM'][var].isel(cell=cells) - LTM['ERA5'][var]
            plt_mesh2 = axs[1].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data.values,
                norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())
            
            cells=get_nn_lon_lat_index(
                hp.get_nside(LTM['ICON']['S']),#2**izlev2,
                LTM['ERA5']['S'].lon.values,
                LTM['ERA5']['S'].lat.values)
            if var in ['S+D']:
                plt_data = (LTM['ICON']['S'] + LTM['ICON']['D']).isel(cell=cells) - (LTM['ERA5']['S'] + LTM['ERA5']['D'])
            elif var in ['S', 'D']:
                plt_data = LTM['ICON'][var].isel(cell=cells) - LTM['ERA5'][var]
            axs[2].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data.values,
                norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())
            
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
                ax=axs, format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.04]))
            cbar.ax.set_xlabel(era5_varlabels[var])
            cbar2 = fig.colorbar(
                plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                ax=axs, format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks2, extend=extend2,
                cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.04]))
            cbar2.ax.set_xlabel(f'Difference in {era5_varlabels[var]}')
        
        fig.subplots_adjust(left=0.001, right=0.999, bottom=fm_bottom, top=0.94)
        fig.savefig(opng)



'''
# check the calculation of omega
pressure = 400
mixing_ratio = mixing_ratio_from_specific_humidity(
    datasets['UM'][f'z{izlev}3H']['hus'].sel(pressure=pressure) * units('kg/kg'))
ds2 = vertical_velocity_pressure(
    datasets['UM'][f'z{izlev}3H']['wa'].sel(pressure=pressure) * units('m/s'),
    pressure * units.hPa,
    datasets['UM'][f'z{izlev}3H']['ta'].sel(pressure=pressure) * units.K,
    mixing_ratio)
print((omega['UM'][f'{pressure}'] == ds2).all().values)

pressure = 400
mixing_ratio = mixing_ratio_from_specific_humidity(
    datasets['ICON'][f'z{izlev}1Dm']['hus'].sel(pressure=pressure * 100) * units('kg/kg'))
ds2 = vertical_velocity_pressure(
        datasets['ICON'][f'z{izlev}1Dm']['wa'].sel(pressure=pressure * 100) * units('m/s'),
        pressure * 100 * units.Pa,
        datasets['ICON'][f'z{izlev}1Dm']['ta'].sel(pressure=pressure * 100) * units.K,
        mixing_ratio)
print((omega['ICON'][f'{pressure}'] == ds2).all().values)


# check the calculation of D
for ids in dss:
    # ids = 'ERA5'
    print(ids)
    omega1 = (omega[ids]['850'] + omega[ids]['700']) / 2
    omega2 = (omega[ids]['600'] + omega[ids]['500'] + omega[ids]['400']) / 3
    dsi = (omega2 - omega1).clip(min=0) * xr.where(-omega1 >= 0, 1, 0) / (-omega2).clip(min=0)
    ds2 = xr.where(np.isinf(dsi), np.nan, dsi).mean(dim='time', skipna=True).compute()
    print((LTM[ids]['D'].values[np.isfinite(LTM[ids]['D'].values)] == ds2.values[np.isfinite(ds2.values)]).all())


'''
# endregion


# region check online data

cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")["online"]
list(cat)
ds = cat["um_glm_n2560_RAL3p3"](time='PT3H').to_dask()


# endregion
