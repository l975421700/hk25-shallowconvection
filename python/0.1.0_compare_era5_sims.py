

# qsub -I -q normal -P nf33 -l walltime=3:00:00,ncpus=1,mem=60GB,jobfs=100MB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46+gdata/qx55


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
from data_process_funcs import (read_MCD06COSP_M3, modis_cmip6_var)
from namelist import cmip6_era5_var, era5_varlabels
import easygems.healpix as egh
import healpy as hp
from matplotlib.colors import BoundaryNorm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as ticker

# endregion


# region plot ERA5, UM, and ICON


dss = ['ERA5', 'UM', 'ICON']
nrow = 1
ncol = len(dss)
fm_bottom = 1.5 / (4.4*nrow + 2)
izlev = 6
izlev2 = 6

datasets = {}
for ids in dss: datasets[ids] = {}

datasets['UM'][f'z{izlev}1H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.z{izlev}.zarr').sel(time=slice('2020-03', '2021-02'))
datasets['ICON'][f'z{izlev2}1Dm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z{izlev2}_atm.zarr').sel(time=slice('2020-03', '2021-02'))


for var2 in ['clivi']:
    # ['rsut', 'clt', 'pr', 'rlut']
    var1 = cmip6_era5_var[var2]
    print(f'#---------------- {var2} {var1}')
    
    if var1 in ['mtuwswrf']:
        era5_mtdwswrf = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtdwswrf/202[0-1]/*.nc')))['mtdwswrf'].sel(time=slice('2020-03', '2021-02'))
        era5_mtnswrf = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/mtnswrf/202[0-1]/*.nc')))['mtnswrf'].sel(time=slice('2020-03', '2021-02'))
        datasets['ERA5'][var2] = (era5_mtnswrf - era5_mtdwswrf).rename({'latitude': 'lat', 'longitude': 'lon'})
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
    
    for imode in ['diff']:
        # ['org', 'diff', 'regrid']
        print(f'#-------------------------------- {imode}')
        
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
            datasets['ERA5'][var2].lon,
            datasets['ERA5'][var2].lat,
            datasets['ERA5'][var2].weighted(datasets['ERA5'][var2].time.dt.days_in_month).mean(dim='time').values,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
        if imode == 'org':
            axs[1].set_global()
            axs[2].set_global()
            
            plt_data = datasets['UM'][f'z{izlev}1H'][var2].mean(dim='time')
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
            
            plt_data = datasets['ICON'][f'z{izlev2}1Dm'][var2].mean(dim='time')
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
            plt_data = datasets['UM'][f'z{izlev}1H'][var2].mean(dim='time').isel(cell=cells)
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
                hp.get_nside(datasets['ICON'][f'z{izlev2}1Dm'][var2]),#2**izlev2,
                datasets['ERA5'][var2].lon.values,
                datasets['ERA5'][var2].lat.values)
            plt_data = datasets['ICON'][f'z{izlev2}1Dm'][var2].mean(dim='time').isel(cell=cells)
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
            plt_data = datasets['UM'][f'z{izlev}1H'][var2].mean(dim='time').isel(cell=cells)
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
                hp.get_nside(datasets['ICON'][f'z{izlev2}1Dm'][var2]),#2**izlev2,
                datasets['ERA5'][var2].lon.values,
                datasets['ERA5'][var2].lat.values)
            plt_data = datasets['ICON'][f'z{izlev2}1Dm'][var2].mean(dim='time').isel(cell=cells)
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


# region plot ERA5 and UM sum


dss = ['ERA5', 'UM']
nrow = 1
ncol = len(dss)
fm_bottom = 1.5 / (4.4*nrow + 2)
izlev = 6

datasets = {}
for ids in dss: datasets[ids] = {}

datasets['UM'][f'z{izlev}1H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.z{izlev}.zarr').sel(time=slice('2020-03', '2021-02'))

datasets['ERA5']['clivi'] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/tciw/202[0-1]/*.nc')))['tciw'].sel(time=slice('2020-03', '2021-02')).rename({'latitude': 'lat', 'longitude': 'lon'})
datasets['ERA5']['clsvi'] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/tcsw/202[0-1]/*.nc')))['tcsw'].sel(time=slice('2020-03', '2021-02')).rename({'latitude': 'lat', 'longitude': 'lon'})

for imode in ['org', 'diff']:
    print(f'#-------------------------------- {imode}')
    
    opng = f"figures/hackathon/0.1.0.0 clivi+clsvi in {', '.join(dss)} {imode}.png"
    
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
        cm_min=0, cm_max=0.3, cm_interval1=0.025, cm_interval2=0.05, cmap='Blues_r',)
    extend = 'max'
    pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        cm_min=-0.1, cm_max=0.3, cm_interval1=0.01, cm_interval2=0.05, cmap='BrBG_r', asymmetric=True)
    extend2 = 'both'
    
    plt_data = (datasets['ERA5']['clivi'] + datasets['ERA5']['clsvi']).weighted(datasets['ERA5']['clivi'].time.dt.days_in_month).mean(dim='time')
    plt_mesh = axs[0].pcolormesh(
        plt_data.lon, plt_data.lat, plt_data.values,
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
    if imode == 'org':
        axs[1].set_global()
        plt_data2 = datasets['UM'][f'z{izlev}1H']['clivi'].mean(dim='time')
        egh.healpix_show(plt_data2, ax=axs[1], norm=pltnorm, cmap=pltcmp)
        
        cbar = fig.colorbar(
            plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
            ax=axs, format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks, extend=extend,
            cax=fig.add_axes([0.3, fm_bottom-0.05, 0.4, 0.04]))
        cbar.ax.set_xlabel(r'total column cloud ice+snow water [$kg \; m^{-2}$]')
    elif imode == 'diff':
        cells=get_nn_lon_lat_index(
            hp.get_nside(datasets['UM'][f'z{izlev}1H']['clivi']),#2**izlev,
            datasets['ERA5']['clivi'].lon.values,
            datasets['ERA5']['clivi'].lat.values)
        plt_data2 = datasets['UM'][f'z{izlev}1H']['clivi'].mean(dim='time').isel(cell=cells)
        plt_mesh2 = axs[1].pcolormesh(
            plt_data2.lon, plt_data2.lat, plt_data2.values - plt_data.values,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())
        
        cbar = fig.colorbar(
            plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
            ax=axs, format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks, extend=extend,
            cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.04]))
        cbar.ax.set_xlabel(r'total column cloud ice+snow water [$kg \; m^{-2}$]')
        cbar2 = fig.colorbar(
            plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
            ax=axs, format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks2, extend=extend2,
            cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.04]))
        cbar2.ax.set_xlabel(r'Difference in total column cloud ice+snow water [$kg \; m^{-2}$]')
    
    fig.subplots_adjust(left=0.001, right=0.999, bottom=fm_bottom, top=0.94)
    fig.savefig(opng)






# endregion


# region plot ERA5 and UM pl


dss = ['ERA5', 'UM']
izlev = 5
nrow = 1
ncol = len(dss)
pwidth  = 8
pheight = 5
fm_bottom = 2.2/(pheight*nrow+2.7)
fm_top = 1 - 0.5/(pheight*nrow+2.7)


datasets = {}
for ids in dss: datasets[ids] = {}

datasets['UM'][f'z{izlev}3H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT3H.z{izlev}.zarr').sel(time=slice('2020-03', '2021-02'))

for var2 in ['qs']:
    # var2 = 'cli'
    var1 = cmip6_era5_var[var2]
    # [['clw', 'cli', 'hur', 'hus', 'ta', 'qr', 'qs']]
    # [['clwc', 'ciwc', 'r', 'q', 't', 'crwc', 'cswc']]
    # [[kg kg-1, kg kg-1, %, kg kg**-1, K]]
    print(f'#-------------------------------- {var2} {var1}')
    
    datasets['ERA5'][var2] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{var1}/202[0-1]/*.nc'))).sel(time=slice('2020-03', '2021-02'))[var1].rename({'latitude': 'lat', 'longitude': 'lon', 'level': 'pressure'})
    cells=get_nn_lon_lat_index(
        hp.get_nside(datasets['UM'][f'z{izlev}3H'][var2][0, 0, :]),#2**izlev2,
        datasets['ERA5'][var2].lon.values,
        datasets['ERA5'][var2].lat.values)
    
    for imode in ['org', 'diff']:
        # imode = 'org'
        print(f'#---------------- {imode}')
        
        opng = f'figures/hackathon/0.1.1.0_{var2} in {', '.join(dss)} {imode} zm.png'
        if imode in ['org', 'regrid']:
            plt_colnames = dss
        elif imode in ['diff']:
            plt_colnames = [dss[0]] + [f'{ids} - {dss[0]}' for ids in dss[1:]]
        
        if var2 in ['clw', 'cli']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=50, cm_interval1=2, cm_interval2=10, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-10, cm_max=50, cm_interval1=2, cm_interval2=10, cmap='BrBG_r', asymmetric=True)
        elif var2 in ['qr']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=20, cm_interval1=1, cm_interval2=2, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG_r')
        elif var2 in ['qs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=30, cm_interval1=1, cm_interval2=3, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-30, cm_max=10, cm_interval1=2, cm_interval2=4, cmap='BrBG_r', asymmetric=True)
        elif var2 in ['hur']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='Blues_r')
            extend = 'neither'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=2.5, cm_interval2=10, cmap='BrBG_r')
        elif var2 in ['hus']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=16, cm_interval1=1, cm_interval2=2, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-3, cm_max=3, cm_interval1=0.25, cm_interval2=0.5, cmap='BrBG_r')
        elif var2 in ['ta']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-60, cm_max=30, cm_interval1=2.5, cm_interval2=10, cmap='PuOr', asymmetric=True)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG_r')
        extend2 = 'both'
        
        fig, axs = plt.subplots(nrow, ncol, figsize=np.array([pwidth*ncol, pheight*nrow+2.7])/2.54, sharey=True, gridspec_kw={'hspace':0.01, 'wspace':0.05})
        
        for jcol in range(ncol):
            axs[jcol].set_xticks(np.arange(-60, 60+1e-4, 30))
            axs[jcol].xaxis.set_minor_locator(ticker.AutoMinorLocator(3))
            axs[jcol].set_xlim(-90, 90)
            axs[jcol].xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
            axs[jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, linestyle='--')
            axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}', ha='left', va='bottom', transform=axs[jcol].transAxes)
        
        plt_data = datasets['ERA5'][var2].mean(dim=['time', 'lon'])
        if var2 in ['hus']: plt_data *= 10**3
        elif var2 in ['clw', 'cli', 'qr', 'qs']: plt_data *= 10**6
        elif var2 in ['ta']: plt_data -= 273.15
        plt_mesh = axs[0].pcolormesh(
            plt_data.lat, plt_data.pressure, plt_data,
            norm=pltnorm, cmap=pltcmp, zorder=1)
        if imode == 'org':
            plt_data2 = datasets['UM'][f'z{izlev}3H'][var2].mean(dim='time').isel(cell=cells).mean(dim='lon')
            if var2 in ['hus']: plt_data2 *= 10**3
            elif var2 in ['clw', 'cli', 'qr', 'qs']:
                plt_data2 *= 10**6
                plt_data2[:] = plt_data2[::-1].values
            elif var2 in ['ta']: plt_data2 -= 273.15
            axs[1].pcolormesh(
                plt_data2.lat, plt_data2.pressure, plt_data2,
                norm=pltnorm, cmap=pltcmp, zorder=1)
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.25, fm_bottom-0.13, 0.5, 0.04]))
            cbar.ax.set_xlabel(f'{era5_varlabels[var1]}')
        elif imode == 'diff':
            plt_data2 = datasets['UM'][f'z{izlev}3H'][var2].mean(dim='time').isel(cell=cells).mean(dim='lon')
            if var2 in ['hus']: plt_data2 *= 10**3
            elif var2 in ['clw', 'cli', 'qr', 'qs']:
                plt_data2 *= 10**6
                plt_data2[:] = plt_data2[::-1].values
            elif var2 in ['ta']: plt_data2 -= 273.15
            plt_mesh2 = axs[1].pcolormesh(
                plt_data2.lat, plt_data2.pressure, plt_data2 - plt_data,
                norm=pltnorm2, cmap=pltcmp2, zorder=1)
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.05, fm_bottom-0.13, 0.4, 0.04]))
            cbar.ax.set_xlabel(f'{era5_varlabels[var1]}')
            cbar2 = fig.colorbar(
                plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks2, extend=extend2,
                cax=fig.add_axes([0.55, fm_bottom-0.13, 0.4, 0.04]))
            cbar2.ax.set_xlabel(f'Difference in {era5_varlabels[var1]}')
        
        axs[0].invert_yaxis()
        axs[0].set_ylim(1000, 0)
        axs[0].set_yticks(np.arange(1000, 0 - 1e-4, -200))
        axs[0].set_ylabel(r'Pressure [$hPa$]')
        fig.subplots_adjust(left=0.1, right=0.999, bottom=fm_bottom, top=fm_top)
        fig.savefig(opng)








'''
    print(f'#---------------- UM')
    print(datasets['UM'][f'z{izlev}3H'][var2])
    print(f'#---------------- ERA5')
    print(datasets['ERA5'][var2])


izlev2 = 6
datasets['ICON'][f'z{izlev2}1Dp'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_point_z{izlev2}_atm.zarr') # no tas/ps # w psl/ta
datasets['ICON'][f'z{izlev2}1Hp'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/PT1H_point_z{izlev2}_atm.zarr') # no tas/ps/ta # w psl
datasets['ICON'][f'z{izlev2}3Hp'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/PT3H_point_z{izlev2}_atm.zarr') # no tas/ps/ta # w psl
datasets['ICON'][f'z{izlev2}3Hm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/PT3H_mean_z{izlev2}_atm.zarr')
datasets['ICON'][f'z{izlev2}6Hp'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/PT6H_point_z{izlev2}_atm.zarr') # no tas/ps/ta # w psl
datasets['ICON'][f'z{izlev2}1Dm'] = xr.open_zarr(f'/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z{izlev2}_atm.zarr').sel(time=slice('2020-03', '2021-02'))

datasets['ICON'][f'z{izlev2}1Dm'][['hur', 'hus']]

for ikey in datasets['ICON'].keys():
    print(ikey)
    if 'clw' in datasets['ICON'][ikey].data_vars:
        print(f'clw in {ikey}')
    if 'cli' in datasets['ICON'][ikey].data_vars:
        print(f'cli in {ikey}')


'''
# endregion


# region plot ERA5 and UM pl sum


dss = ['ERA5', 'UM']
izlev = 5
nrow = 1
ncol = len(dss)
pwidth  = 8
pheight = 5
fm_bottom = 2.2/(pheight*nrow+2.7)
fm_top = 1 - 0.5/(pheight*nrow+2.7)

datasets = {}
for ids in dss: datasets[ids] = {}

datasets['UM'][f'z{izlev}3H'] = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT3H.z{izlev}.zarr').sel(time=slice('2020-03', '2021-02'))

# vars = ['cli', 'qs']
# cbar_label = r'cloud ice+snow water content [$10^{-3} \; g \; kg^{-1}$]'
vars = ['clw', 'qr']
cbar_label = r'cloud liquid+rain water content [$10^{-3} \; g \; kg^{-1}$]'

for var2 in vars:
    # var2 = 'cli'
    var1 = cmip6_era5_var[var2]
    # [['clw', 'cli', 'hur', 'hus', 'ta', 'qr', 'qs']]
    # [['clwc', 'ciwc', 'r', 'q', 't', 'crwc', 'cswc']]
    # [[kg kg-1, kg kg-1, %, kg kg**-1, K]]
    print(f'#-------------------------------- {var2} {var1}')
    
    datasets['ERA5'][var2] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{var1}/202[0-1]/*.nc'))).sel(time=slice('2020-03', '2021-02'))[var1].rename({'latitude': 'lat', 'longitude': 'lon', 'level': 'pressure'})

cells=get_nn_lon_lat_index(
    hp.get_nside(datasets['UM'][f'z{izlev}3H'][var2][0, 0, :]),#2**izlev2,
    datasets['ERA5'][var2].lon.values,
    datasets['ERA5'][var2].lat.values)

for imode in ['org']:
    # imode = 'org'
    print(f'#---------------- {imode}')
    
    opng = f'figures/hackathon/0.1.1.0_{'+'.join(vars)} in {', '.join(dss)} {imode} zm.png'
    if imode in ['org', 'regrid']:
        plt_colnames = dss
    elif imode in ['diff']:
        plt_colnames = [dss[0]] + [f'{ids} - {dss[0]}' for ids in dss[1:]]
        
    if vars in [['cli', 'qs'], ['clw', 'qr']]:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=50, cm_interval1=2, cm_interval2=10, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-20, cm_max=40, cm_interval1=2, cm_interval2=10, cmap='BrBG_r', asymmetric=True)
    extend2 = 'both'
    
    fig, axs = plt.subplots(nrow, ncol, figsize=np.array([pwidth*ncol, pheight*nrow+2.7])/2.54, sharey=True, gridspec_kw={'hspace':0.01, 'wspace':0.05})
    
    for jcol in range(ncol):
        axs[jcol].set_xticks(np.arange(-60, 60+1e-4, 30))
        axs[jcol].xaxis.set_minor_locator(ticker.AutoMinorLocator(3))
        axs[jcol].set_xlim(-90, 90)
        axs[jcol].xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
        axs[jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, linestyle='--')
        axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}', ha='left', va='bottom', transform=axs[jcol].transAxes)
    
    plt_data = sum(datasets['ERA5'][var2] for var2 in vars).mean(dim=['time', 'lon'])
    if vars in [['cli', 'qs'], ['clw', 'qr']]: plt_data *= 10**6
    plt_mesh = axs[0].pcolormesh(
        plt_data.lat, plt_data.pressure, plt_data,
        norm=pltnorm, cmap=pltcmp, zorder=1)
    if imode == 'org':
        plt_data2 = sum(datasets['UM'][f'z{izlev}3H'][var2] for var2 in vars).mean(dim='time').isel(cell=cells).mean(dim='lon')
        if vars in [['cli', 'qs'], ['clw', 'qr']]:
            plt_data2 *= 10**6
            plt_data2[:] = plt_data2[::-1].values
        axs[1].pcolormesh(
            plt_data2.lat, plt_data2.pressure, plt_data2,
            norm=pltnorm, cmap=pltcmp, zorder=1)
        cbar = fig.colorbar(
            plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
            format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks, extend=extend,
            cax=fig.add_axes([0.25, fm_bottom-0.13, 0.5, 0.04]))
        cbar.ax.set_xlabel(f'{cbar_label}')
    elif imode == 'diff':
        plt_data2 = sum(datasets['UM'][f'z{izlev}3H'][var2] for var2 in vars).mean(dim='time').isel(cell=cells).mean(dim='lon')
        if vars in [['cli', 'qs'], ['clw', 'qr']]:
            plt_data2 *= 10**6
            plt_data2[:] = plt_data2[::-1].values
        plt_mesh2 = axs[1].pcolormesh(
                plt_data2.lat, plt_data2.pressure, plt_data2 - plt_data,
                norm=pltnorm2, cmap=pltcmp2, zorder=1)
        cbar = fig.colorbar(
            plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
            format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks, extend=extend,
            cax=fig.add_axes([0.05, fm_bottom-0.13, 0.4, 0.04]))
        cbar.ax.set_xlabel(f'{cbar_label}')
        cbar2 = fig.colorbar(
            plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
            format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks2, extend=extend2,
            cax=fig.add_axes([0.55, fm_bottom-0.13, 0.4, 0.04]))
        cbar2.ax.set_xlabel(f'Difference in {cbar_label}')
    
    axs[0].invert_yaxis()
    axs[0].set_ylim(1000, 0)
    axs[0].set_yticks(np.arange(1000, 0 - 1e-4, -200))
    axs[0].set_ylabel(r'Pressure [$hPa$]')
    fig.subplots_adjust(left=0.1, right=0.999, bottom=fm_bottom, top=fm_top)
    fig.savefig(opng)




# endregion


