import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import string
from matplotlib import cm
import os
import sys
sys.path.append('/home/548/cd3022/hk25-shallowconvection/module')
from plot_funcs import (globe_plot, remove_trailing_zero_pos, plt_mesh_pars)
from pathlib import Path
import xesmf as xe
import easygems.healpix as egh
import healpy as hp

def ceres_diff(figname, variable, cbar_range1, cbar_range2):
    ##### CERES Data
    ceres = xr.open_dataset('/g/data/er8/users/cd3022/hk25-ShallowConvection/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202502.nc')
    ceres['toa_sw_all_mon'] = ceres['toa_sw_all_mon'] * -1
    ceres['toa_lw_all_mon'] = ceres['toa_lw_all_mon'] * -1
    ceres = ceres.rename({'toa_sw_all_mon': 'rsut'})
    ceres = ceres.rename({'toa_lw_all_mon': 'rlut'})
    ceres = ceres.sel(time=slice('2020-03','2021-02'))
    ceres_data = ceres[variable].mean(dim='time')

    ##### ERA5 Data
    era5_dirs = [
        Path("/g/data/rt52/era5/single-levels/monthly-averaged/mtdwswrf/2020"),
        Path("/g/data/rt52/era5/single-levels/monthly-averaged/mtdwswrf/2021"),
        Path("/g/data/rt52/era5/single-levels/monthly-averaged/mtnswrf/2020"),
        Path("/g/data/rt52/era5/single-levels/monthly-averaged/mtnswrf/2021"),
        Path("/g/data/rt52/era5/single-levels/monthly-averaged/mtnlwrf/2020"),
        Path("/g/data/rt52/era5/single-levels/monthly-averaged/mtnlwrf/2021"),
    ]
    era5_files = sorted([f for d in era5_dirs for f in d.glob("*.nc")])
    era5 = xr.open_mfdataset(
        era5_files,
        combine='by_coords',   # Use coordinate-based merging
        data_vars='minimal',   # Avoid trying to align data_vars across files
        coords='minimal',      # Avoid "coords='different'" error
        compat='override',     # Avoid unnecessary compatibility checks
    )
    era5 = era5.sel(time=slice('2020-03','2021-02'))
    era5['rsut'] = era5['mtnswrf'] - era5['mtdwswrf']
    era5 = era5.rename({'mtnlwrf':'rlut'})
    era5_data = era5[variable].mean(dim='time')

    # regrid to ceres grid
    era5_data = era5_data.rename({'latitude': 'lat', 'longitude': 'lon'})
    era5_data = era5_data.sortby(['lat', 'lon'])
    regridder = xe.Regridder(era5_data, ceres_data, method='bilinear', periodic=True)
    era5_data = regridder(era5_data)

    # era5 ceres DIFF

    era5_ceres_diff = era5_data - ceres_data
    era5_ceres_rmse = np.sqrt(np.square(era5_ceres_diff).weighted(np.cos(np.deg2rad(era5_ceres_diff.lat))).mean()).values

    ##### UK node model data #####

    um = xr.open_zarr('/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.z5.zarr')
    um_data = um[variable].sel(time=slice('2020-03', '2021-02')).mean(dim='time') *-1

    # Regrid healpix to era5

    def get_nn_lon_lat_index(nside, lons, lats):
        """
        nside: integer, power of 2. The return of hp.get_nside()
        lons: uniques values of longitudes
        lats: uniques values of latitudes
        returns: array with the HEALPix cells that are closest to the lon/lat grid
        """
        lons2, lats2 = np.meshgrid(lons, lats)
        return xr.DataArray(
            hp.ang2pix(nside, lons2, lats2, nest = True, lonlat = True),
            coords=[("lat", lats), ("lon", lons)],
        )
    # Find the HEALPix pixels that are closest to the ERA5 grid

    # longitudes and latitudes for the ERA5 grid
    lon = ceres['lon'].values
    lat = ceres['lat'].values
    print(f'lon shape: {lon.shape}')
    print(f'lat shape: {lat.shape}')

    # nside for um simulation, it should be equal to 2**zoom
    this_nside = hp.get_nside(um_data)
    print(f'this_nside shape: {this_nside.shape}')

    cells = get_nn_lon_lat_index(this_nside, lon, lat)
    print(f'cells shape: {cells.shape}')

    # Calculate UM-CERES difference

    um_ceres_diff = um_data.isel(cell = cells) - ceres_data
    um_ceres_rmse = np.sqrt(np.square(um_ceres_diff).weighted(np.cos(np.deg2rad(um_ceres_diff.lat))).mean()).values
    print(f'um_data shape: {um_data.shape}')
    print(f'um_ceres_diff shape: {um_ceres_diff.shape}')
    ##### GER node model data #####

    icon = xr.open_zarr('/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z5_atm.zarr')
    icon_data = icon[variable].sel(time=slice('2020-03', '2021-02')).mean(dim='time') *-1

    # Regrid healpix to era5
    # Find the HEALPix pixels that are closest to the ERA5 grid

    # longitudes and latitudes for the ERA5 grid
    lon = ceres['lon'].values
    lat = ceres['lat'].values

    # nside for um simulation, it should be equal to 2**zoom
    this_nside = hp.get_nside(icon_data)

    cells = get_nn_lon_lat_index(this_nside, lon, lat)

    # Calculate ICON-CERES difference

    icon_ceres_diff = icon_data.isel(cell = cells) - ceres_data
    icon_ceres_rmse = np.sqrt(np.square(icon_ceres_diff).weighted(np.cos(np.deg2rad(icon_ceres_diff.lat))).mean()).values


    ##### Plotting #####
    nrow = 1
    ncol = 4
    fm_bottom = 1.5 / (4.4*nrow + 2)
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
        subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

    plt_colnames = [
        'CERES',
        f'ERA5-CERES, RMSE: {era5_ceres_rmse:.2f}',
        f'UM-CERES, RMSE: {um_ceres_rmse:.2f}',
        f'ICON-CERES, RMSE: {icon_ceres_rmse:.2f}',
        ]
    for jcol in range(ncol):
        axs[jcol] = globe_plot(ax_org=axs[jcol])
        axs[jcol].text(
            0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
            ha='left', va='bottom', transform=axs[jcol].transAxes)

    # example to add colorbar
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=cbar_range1[0], cm_max=cbar_range1[1], cm_interval1=10, cm_interval2=20, cmap='viridis',)
    pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        cm_min=cbar_range2[0], cm_max=cbar_range2[1], cm_interval1=5, cm_interval2=10, cmap='BrBG')

    axs[0].pcolormesh(
            ceres.lon, ceres.lat,
            ceres_data,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())

    axs[1].pcolormesh(
            era5_ceres_diff.lon, era5_ceres_diff.lat,
            era5_ceres_diff,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())

    axs[2].pcolormesh(
            um_ceres_diff.lon, um_ceres_diff.lat,
            um_ceres_diff,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())

    axs[3].pcolormesh(
            icon_ceres_diff.lon, icon_ceres_diff.lat,
            icon_ceres_diff,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())

    cbar = fig.colorbar(
        cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #plt_mesh, #
        ax=axs, format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks, extend='both',
        cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.04]))
    cbar.ax.set_xlabel(f'{variable} (W/m-2)')
    cbar2 = fig.colorbar(
        cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #plt_mesh2, #
        ax=axs, format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend='both',
        cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.04]))
    cbar2.ax.set_xlabel(f'{variable} bias (W/m-2)')

    fig.subplots_adjust(left=0.001, right=0.999, bottom=fm_bottom, top=0.94)
    # fig.savefig(F'figures/{figname}.png')
    return
ceres_diff(figname='lw_ceres_bias', variable='rlut', cbar_range1=(-300, -30), cbar_range2=(-40, 40))
'''
variables:
rsut: toa outgoing shortwave
rlut: toa outgoing longwave
'''


um = xr.open_zarr('/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.z5.zarr')
um.variables

ceres = xr.open_dataset('/g/data/er8/users/cd3022/hk25-ShallowConvection/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202502.nc')
ceres.variables