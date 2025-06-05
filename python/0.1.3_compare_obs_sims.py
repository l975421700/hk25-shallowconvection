

# qsub -I -q normal -P nf33 -l walltime=1:00:00,ncpus=1,mem=15GB,jobfs=100MB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46+gdata/qx55


# region import packages


import xarray as xr
import glob
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import string
import numpy as np
import os
import sys
sys.path.append(os.getcwd() + '/code/hk25-shallowconvection/module')
from plot_funcs import (globe_plot, remove_trailing_zero_pos, plt_mesh_pars, get_nn_lon_lat_index)
from data_process_funcs import (read_MCD06COSP_M3, modis_cmip6_var, regrid)
from namelist import cmip6_era5_var, era5_varlabels
import easygems.healpix as egh
import healpy as hp
from matplotlib.colors import BoundaryNorm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as ticker

# endregion


# region plot obs and sim

dss = ['CERES', 'ERA5', 'UM', 'ICON']
nrow = 1
ncol = len(dss)
fm_bottom = 1.5 / (4.4*nrow + 2)
izlev = 6
izlev2 = 6

datasets = {}
for ids in dss: datasets[ids] = {}

datasets['UM'][f'z{izlev}1H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.z{izlev}.zarr').sel(time=slice('2020-03', '2021-02'))
datasets['ICON'][f'z{izlev2}1Dm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z{izlev2}_atm.zarr').sel(time=slice('2020-03', '2021-02'))

datasets['UM'][f'z{izlev}1H']['rsutcl'] = datasets['UM'][f'z{izlev}1H']['rsut'] - datasets['UM'][f'z{izlev}1H']['rsutcs']
datasets['ICON'][f'z{izlev2}1Dm']['rsutcl'] = datasets['ICON'][f'z{izlev2}1Dm']['rsut'] - datasets['ICON'][f'z{izlev2}1Dm']['rsutcs']

datasets['UM'][f'z{izlev}1H']['rlutcl'] = datasets['UM'][f'z{izlev}1H']['rlut'] - datasets['UM'][f'z{izlev}1H']['rlutcs']
datasets['ICON'][f'z{izlev2}1Dm']['rlutcl'] = datasets['ICON'][f'z{izlev2}1Dm']['rlut'] - datasets['ICON'][f'z{izlev2}1Dm']['rlutcs']

if 'CERES' in dss:
    datasets['CERES'] = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice('2020-03', '2021-02'))
    datasets['CERES'] = datasets['CERES'].rename({
        'toa_sw_all_mon': 'mtuwswrf',
        'toa_lw_all_mon': 'mtnlwrf',
        'solar_mon': 'mtdwswrf',
        'toa_sw_clr_c_mon': 'mtuwswrfcs',
        'toa_lw_clr_c_mon': 'mtnlwrfcs',
        })
    datasets['CERES']['mtuwswrf'] *= (-1)
    datasets['CERES']['mtnlwrf'] *= (-1)
    datasets['CERES']['mtuwswrfcs'] *= (-1)
    datasets['CERES']['mtnlwrfcs'] *= (-1)
    
    datasets['CERES']['mtuwswrfcl'] = datasets['CERES']['mtuwswrf'] - datasets['CERES']['mtuwswrfcs']
    datasets['CERES']['mtnlwrfcl'] = datasets['CERES']['mtnlwrf'] - datasets['CERES']['mtnlwrfcs']

def std_units(ds, var):
    if var in ['pr', 'evspsbl', 'evspsblpot']:
        ds *= 86400
    elif var in ['tas', 'ts']:
        ds -= 273.15
    elif var in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rlutcl', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'rsutcl', 'hfls', 'hfss']:
        ds *= (-1)
    elif var in ['psl']:
        ds /= 100
    elif var in ['huss']:
        ds *= 1000
    elif var in ['clt']:
        ds *= 100
    
    return(ds)


for var2 in ['rsutcs']:
    # ['rlut', 'rsut', 'rsdt', 'rlutcl', 'rsutcl', 'rlutcs', 'rsutcs']
    # var2 = 'rsut'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var2} {var1}')
    
    if var1 in ['mtuwswrf']:
        era5_mtdwswrf = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtdwswrf/202[0-1]/*.nc')))['mtdwswrf'].sel(time=slice('2020-03', '2021-02'))
        era5_mtnswrf = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtnswrf/202[0-1]/*.nc')))['mtnswrf'].sel(time=slice('2020-03', '2021-02'))
        datasets['ERA5'][var2] = (era5_mtnswrf - era5_mtdwswrf).rename({'latitude': 'lat', 'longitude': 'lon'})
        del era5_mtdwswrf, era5_mtnswrf
    elif var1 in ['mtuwswrfcs']:
        era5_mtdwswrf = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtdwswrf/202[0-1]/*.nc')))['mtdwswrf'].sel(time=slice('2020-03', '2021-02'))
        era5_mtnswrfcs = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtnswrfcs/202[0-1]/*.nc')))['mtnswrfcs'].sel(time=slice('2020-03', '2021-02'))
        datasets['ERA5'][var2] = (era5_mtnswrfcs - era5_mtdwswrf).rename({'latitude': 'lat', 'longitude': 'lon'})
        del era5_mtdwswrf, era5_mtnswrfcs
    elif var1 in ['mtuwswrfcl']:
        era5_mtnswrf = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtnswrf/202[0-1]/*.nc')))['mtnswrf'].sel(time=slice('2020-03', '2021-02'))
        era5_mtnswrfcs = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtnswrfcs/202[0-1]/*.nc')))['mtnswrfcs'].sel(time=slice('2020-03', '2021-02'))
        datasets['ERA5'][var2] = (era5_mtnswrf - era5_mtnswrfcs).rename({'latitude': 'lat', 'longitude': 'lon'})
        del era5_mtnswrf, era5_mtnswrfcs
    elif var1 in ['mtnlwrfcl']:
        era5_mtnlwrf = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtnlwrf/202[0-1]/*.nc')))['mtnlwrf'].sel(time=slice('2020-03', '2021-02'))
        era5_mtnlwrfcs = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtnlwrfcs/202[0-1]/*.nc')))['mtnlwrfcs'].sel(time=slice('2020-03', '2021-02'))
        datasets['ERA5'][var2] = (era5_mtnlwrf - era5_mtnlwrfcs).rename({'latitude': 'lat', 'longitude': 'lon'})
        del era5_mtnlwrf, era5_mtnlwrfcs
    else:
        datasets['ERA5'][var2] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/{var1}/202[0-1]/*.nc')))[var1].sel(time=slice('2020-03', '2021-02')).rename({'latitude': 'lat', 'longitude': 'lon'})
    if var1 in ['tp', 'e', 'cp', 'lsp', 'pev']:
        datasets['ERA5'][var2] *= 1000
    elif var1 in ['msl']:
        datasets['ERA5'][var2] /= 100
    elif var1 in ['sst', 't2m', 'd2m', 'skt']:
        datasets['ERA5'][var2] -= 273.15
    elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
        datasets['ERA5'][var2] *= 100
    elif var1 in ['z']:
        datasets['ERA5'][var2] /= 9.80665
    elif var1 in ['mper']:
        datasets['ERA5'][var2] *= 86400
    
    for imode in ['diff']:
        # ['org', 'diff', 'regrid']
        # imode='org'
        print(f'#---------------- {imode}')
        
        opng = f"figures/hackathon/0.1.0.0 {var2} in {', '.join(dss)} {imode}.png"
        if imode in ['org', 'regrid']:
            plt_colnames = dss
        elif imode in ['diff']:
            plt_colnames = [dss[0]] + [f'{ids} - {dss[0]}' for ids in dss[1:]]
        
        fig, axs = plt.subplots(
            nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
            subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
            gridspec_kw={'hspace': 0.01, 'wspace': 0.01})
        
        for jcol in range(ncol):
            axs[jcol] = globe_plot(ax_org=axs[jcol])
            axs[jcol].text(
                0,1.02,f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                ha='left', va='bottom', transform=axs[jcol].transAxes)
        
        if var2 in ['clt']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='Blues_r')
            extend = 'neither'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['rsut']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-250, cm_max=-20, cm_interval1=10, cm_interval2=20, cmap='viridis')
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['rsutcl']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-100, cm_max=0, cm_interval1=10, cm_interval2=10, cmap='viridis')
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['rsdt']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=200, cm_max=400, cm_interval1=10, cm_interval2=20, cmap='viridis_r')
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['rsutcs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-100, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis')
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['pr']:
            pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
            pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
            pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
            pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
            extend = 'max'
            pltlevel2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
            pltticks2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
            pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
            pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
            extend2 = 'both'
        elif var2 in ['rlut', 'rlutcs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-300, cm_max=-130, cm_interval1=10, cm_interval2=20, cmap='viridis',)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['rlutcl']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-20, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='PRGn', asymmetric=True)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['prw']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['clwvi']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=0.3, cm_interval1=0.025, cm_interval2=0.05, cmap='Blues_r',)
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-0.1, cm_max=0.1, cm_interval1=0.02, cm_interval2=0.02, cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['clivi']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=0.3, cm_interval1=0.025, cm_interval2=0.05, cmap='Blues_r',)
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-0.1, cm_max=0.3, cm_interval1=0.01, cm_interval2=0.05, cmap='BrBG_r', asymmetric=True)
            extend2 = 'both'
        
        plt_mesh = axs[0].pcolormesh(
            datasets['CERES'][var1].lon,
            datasets['CERES'][var1].lat,
            datasets['CERES'][var1].weighted(datasets['CERES'][var1].time.dt.days_in_month).mean(dim='time').values,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
        
        if imode == 'org':
            axs[1].pcolormesh(
                datasets['ERA5'][var2].lon,
                datasets['ERA5'][var2].lat,
                datasets['ERA5'][var2].weighted(datasets['ERA5'][var2].time.dt.days_in_month).mean(dim='time').values,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
            
            axs[2].set_global()
            axs[3].set_global()
            plt_data = datasets['UM'][f'z{izlev}1H'][var2].mean(dim='time')
            plt_data = std_units(plt_data, var=var2)
            egh.healpix_show(plt_data, ax=axs[2], norm=pltnorm, cmap=pltcmp)
            
            plt_data = datasets['ICON'][f'z{izlev2}1Dm'][var2].mean(dim='time')
            plt_data = std_units(plt_data, var=var2)
            egh.healpix_show(plt_data, ax=axs[3], norm=pltnorm, cmap=pltcmp)
        elif imode in ['diff']:
            plt_data = regrid(datasets['ERA5'][var2].weighted(datasets['ERA5'][var2].time.dt.days_in_month).mean(dim='time'), ds_out=datasets['CERES'][var1]) - datasets['CERES'][var1].weighted(datasets['CERES'][var1].time.dt.days_in_month).mean(dim='time').values
            axs[1].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data.values,
                norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())
            
            cells=get_nn_lon_lat_index(
                2**izlev,
                datasets['CERES'][var1].lon.values,
                datasets['CERES'][var1].lat.values)
            plt_data = datasets['UM'][f'z{izlev}1H'][var2].mean(dim='time').isel(cell=cells)
            plt_data = std_units(plt_data, var=var2)
            plt_mesh2 = axs[2].pcolormesh(
                plt_data.lon, plt_data.lat,
                plt_data.values - datasets['CERES'][var1].weighted(datasets['CERES'][var1].time.dt.days_in_month).mean(dim='time').values,
                norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())
            
            cells=get_nn_lon_lat_index(
                2**izlev2,
                datasets['CERES'][var1].lon.values,
                datasets['CERES'][var1].lat.values)
            plt_data = datasets['ICON'][f'z{izlev2}1Dm'][var2].mean(dim='time').isel(cell=cells)
            plt_data = std_units(plt_data, var=var2)
            axs[3].pcolormesh(
                plt_data.lon, plt_data.lat,
                plt_data.values - datasets['CERES'][var1].weighted(datasets['CERES'][var1].time.dt.days_in_month).mean(dim='time').values,
                norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())
            
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
                ax=axs, format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.04]))
            cbar.ax.set_xlabel(era5_varlabels[var1])
            cbar2 = fig.colorbar(
                plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                ax=axs, format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks2, extend=extend2,
                cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.04]))
            cbar2.ax.set_xlabel(f'Difference in {era5_varlabels[var1]}')
        
        if imode in ['org', 'regrid']:
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
                ax=axs, format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.3, fm_bottom-0.05, 0.4, 0.04]))
            cbar.ax.set_xlabel(era5_varlabels[var1])
        
        fig.subplots_adjust(left=0.001, right=0.999, bottom=fm_bottom, top=0.94)
        fig.savefig(opng)
    
    del datasets['ERA5'][var2]






# endregion


