

# qsub -I -q normal -l walltime=1:00:00,ncpus=1,mem=60GB,jobfs=100MB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46


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


# region plot ERA5, ICON, and UM


def get_nn_lon_lat_index(nside, lons, lats):
    import xarray as xr
    import numpy as np
    import healpy as hp
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        hp.ang2pix(nside, lons2, lats2, nest=True, lonlat=True),
        coords=[("lat", lats), ("lon", lons)],
    )


dss = ['ERA5', 'UM', 'ICON']
nrow = 1
ncol = len(dss)
fm_bottom = 1.5 / (4.4*nrow + 2)
izlev = 5

datasets = {}
for ids in dss: datasets[ids] = {}

datasets['UM'][f'z{izlev}1H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.z{izlev}.zarr')
datasets['ICON'][f'z{izlev}1Dm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z{izlev}_atm.zarr')


for imode in ['org', 'diff']:
    # ['org', 'diff', 'regrid']
    print(f'#-------------------------------- {imode}')
    for var2 in ['clwvi', 'clivi', 'prw']:
        # ['rsut', 'clt', 'pr', 'rlut']
        var1 = cmip6_era5_var[var2]
        print(f'#---------------- {var2} {var1}')
        
        opng = f"figures/hackathon/0.1.0.0 {var2} in {', '.join(dss)} {imode}.png"
        
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
        elif var2 in ['rlut']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-300, cm_max=-130, cm_interval1=10, cm_interval2=20, cmap='viridis',)
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
                cm_min=0, cm_max=0.5, cm_interval1=0.05, cm_interval2=0.05, cmap='Blues_r',)
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-0.1, cm_max=0.1, cm_interval1=0.2, cm_interval2=0.2, cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['clivi']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.1, cmap='Blues_r',)
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-0.4, cm_max=0.4, cm_interval1=0.1, cm_interval2=0.1, cmap='BrBG_r')
            extend2 = 'both'
        
        if var1 in ['mtuwswrf']:
            era5_mtdwswrf = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtdwswrf/202[0-1]/*.nc')))['mtdwswrf']
            era5_mtnswrf = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtnswrf/202[0-1]/*.nc')))['mtnswrf']
            datasets['ERA5'][var2] = (era5_mtnswrf - era5_mtdwswrf).sel(time=slice('2020-03', '2021-02')).rename({'latitude': 'lat', 'longitude': 'lon'})
            del era5_mtdwswrf, era5_mtnswrf
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
        
        plt_mesh = axs[0].pcolormesh(
            datasets['ERA5'][var2].lon,
            datasets['ERA5'][var2].lat,
            datasets['ERA5'][var2].weighted(datasets['ERA5'][var2].time.dt.days_in_month).mean(dim='time').values,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
        if imode == 'org':
            axs[1].set_global()
            axs[2].set_global()
            
            plt_data = datasets['UM'][f'z{izlev}1H'][var2].sel(time=slice('2020-03', '2021-02')).mean(dim='time')
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                plt_data *= 86400
            elif var2 in ['tas', 'ts']:
                plt_data -= 273.15
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                plt_data *= (-1)
            elif var2 in ['psl']:
                plt_data /= 100
            elif var2 in ['huss']:
                plt_data *= 1000
            elif var2 in ['clt']:
                plt_data *= 100
            egh.healpix_show(plt_data, ax=axs[1], norm=pltnorm, cmap=pltcmp)
            
            plt_data = datasets['ICON'][f'z{izlev}1Dm'][var2].sel(time=slice('2020-03', '2021-02')).mean(dim='time')
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                plt_data *= 86400
            elif var2 in ['tas', 'ts']:
                plt_data -= 273.15
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                plt_data *= (-1)
            elif var2 in ['psl']:
                plt_data /= 100
            elif var2 in ['huss']:
                plt_data *= 1000
            elif var2 in ['clt']:
                plt_data *= 100
            egh.healpix_show(plt_data, ax=axs[2], norm=pltnorm, cmap=pltcmp)
        elif imode == 'regrid':
            cells=get_nn_lon_lat_index(
                hp.get_nside(datasets['UM'][f'z{izlev}1H'][var2]),#2**izlev,
                datasets['ERA5'][var2].lon.values,
                datasets['ERA5'][var2].lat.values)
            plt_data = datasets['UM'][f'z{izlev}1H'][var2].sel(time=slice('2020-03', '2021-02')).mean(dim='time').isel(cell=cells)
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                plt_data *= 86400
            elif var2 in ['tas', 'ts']:
                plt_data -= 273.15
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                plt_data *= (-1)
            elif var2 in ['psl']:
                plt_data /= 100
            elif var2 in ['huss']:
                plt_data *= 1000
            elif var2 in ['clt']:
                plt_data *= 100
            axs[1].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data.values,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
            
            cells=get_nn_lon_lat_index(
                hp.get_nside(datasets['ICON'][f'z{izlev}1Dm'][var2]),#2**izlev,
                datasets['ERA5'][var2].lon.values,
                datasets['ERA5'][var2].lat.values)
            plt_data = datasets['ICON'][f'z{izlev}1Dm'][var2].sel(time=slice('2020-03', '2021-02')).mean(dim='time').isel(cell=cells)
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                plt_data *= 86400
            elif var2 in ['tas', 'ts']:
                plt_data -= 273.15
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                plt_data *= (-1)
            elif var2 in ['psl']:
                plt_data /= 100
            elif var2 in ['huss']:
                plt_data *= 1000
            elif var2 in ['clt']:
                plt_data *= 100
            axs[2].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data.values,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
        elif imode in ['diff']:
            cells=get_nn_lon_lat_index(
                hp.get_nside(datasets['UM'][f'z{izlev}1H'][var2]),#2**izlev,
                datasets['ERA5'][var2].lon.values,
                datasets['ERA5'][var2].lat.values)
            plt_data = datasets['UM'][f'z{izlev}1H'][var2].sel(time=slice('2020-03', '2021-02')).mean(dim='time').isel(cell=cells)
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                plt_data *= 86400
            elif var2 in ['tas', 'ts']:
                plt_data -= 273.15
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                plt_data *= (-1)
            elif var2 in ['psl']:
                plt_data /= 100
            elif var2 in ['huss']:
                plt_data *= 1000
            elif var2 in ['clt']:
                plt_data *= 100
            plt_mesh2 = axs[1].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data.values - datasets['ERA5'][var2].weighted(datasets['ERA5'][var2].time.dt.days_in_month).mean(dim='time').values,
                norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())
            
            cells=get_nn_lon_lat_index(
                hp.get_nside(datasets['ICON'][f'z{izlev}1Dm'][var2]),#2**izlev,
                datasets['ERA5'][var2].lon.values,
                datasets['ERA5'][var2].lat.values)
            plt_data = datasets['ICON'][f'z{izlev}1Dm'][var2].sel(time=slice('2020-03', '2021-02')).mean(dim='time').isel(cell=cells)
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                plt_data *= 86400
            elif var2 in ['tas', 'ts']:
                plt_data -= 273.15
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                plt_data *= (-1)
            elif var2 in ['psl']:
                plt_data /= 100
            elif var2 in ['huss']:
                plt_data *= 1000
            elif var2 in ['clt']:
                plt_data *= 100
            axs[2].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data.values - datasets['ERA5'][var2].weighted(datasets['ERA5'][var2].time.dt.days_in_month).mean(dim='time').values,
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

