import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import string
import xarray as xr
import easygems.healpix as egh
import healpy as hp
from pathlib import Path
import os
import sys
sys.path.append('/home/548/cd3022/hk25-shallowconvection/module')
from plot_funcs import (globe_plot, remove_trailing_zero_pos, plt_mesh_pars)
import metpy

# qsub -I -q normal -P nf33 -l walltime=2:00:00,ncpus=1,mem=120GB,jobfs=100MB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46+gdata/qx55+gdata/hh5
# source /scratch/nf33/public/hackathon_env/bin/activate

um = xr.open_zarr('/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT3H.z9.zarr') # data is already stored in chunks
icon = xr.open_zarr('/g/data/qx55/germany_node/d3hp003.zarr/P1D_mean_z9_atm.zarr')
icon['pressure'] = icon['pressure'] / 100
icon['pressure'].attrs['units'] = 'hPa'

era5_dirs = [
    # Path('/g/data/rt52/era5/pressure-levels/reanalysis/r/2020'),
    # Path('/g/data/rt52/era5/pressure-levels/reanalysis/r/2021'),
    # Path('/g/data/rt52/era5/pressure-levels/reanalysis/t/2020'),
    # Path('/g/data/rt52/era5/pressure-levels/reanalysis/t/2021'),
    Path('/g/data/rt52/era5/pressure-levels/reanalysis/w/2020'),
    Path('/g/data/rt52/era5/pressure-levels/reanalysis/w/2021'),
]
# import glob
# sorted(glob.glob('/g/data/rt52/era5/pressure-levels/reanalysis/w/202[0-1]/*.nc'))
era5_files = sorted([f for d in era5_dirs for f in d.glob("*.nc")])
era5 = xr.open_mfdataset(
    era5_files,
    combine='by_coords',   # Use coordinate-based merging
    data_vars='minimal',   # Avoid trying to align data_vars across files
    coords='minimal',      # Avoid "coords='different'" error
    compat='override',     # Avoid unnecessary compatibility checks
    # parallel=True
)
era5 = era5.sel(time=slice('2020-03', '2021-02'))
era5 = era5.rename({'latitude': 'lat', 'longitude': 'lon'})


##### s index calculations, from Sherwood et al. (2014) #####
s_index_um = (
    (((um.hur.sel(pressure=700) - um.hur.sel(pressure=850)) / 100) - ((um.ta.sel(pressure=700) - um.ta.sel(pressure=850)) / 9)) / 2
    ).mean(dim='time')

s_index_icon = (
    (((icon.hur.sel(pressure=70000) - icon.hur.sel(pressure=85000)) / 100) - ((icon.ta.sel(pressure=70000) - icon.ta.sel(pressure=85000)) / 9)) / 2
    ).mean(dim='time')


s_index_era5 = (
    (((era5.r.sel(level=700) - era5.r.sel(level=850)) / 100) - ((era5.t.sel(level=700) - era5.t.sel(level=850)) / 9)) / 2
).mean(dim='time')

##### D index calculations, from Sherwood et al. (2014) #####
def calc_omega(ds, pressure): 
    ds['mixing_ratio'] = metpy.calc.mixing_ratio_from_specific_humidity(ds.hus)   
    return metpy.calc.vertical_velocity_pressure(
        w = ds.wa.sel(pressure=pressure),
        pressure = ds.pressure.sel(pressure=pressure),
        temperature = ds.ta.sel(pressure=pressure),
        mixing_ratio = ds.mixing_ratio.sel(pressure=pressure)
        )

def calc_d_index(ds):
    if 'w' in ds:
        omega1 = (ds.w.sel(level=850) + ds.w.sel(level=700)) / 2
        omega2 = (ds.w.sel(level=600) + ds.w.sel(level=500) + ds.w.sel(level=400)) / 3
    else:

        omegas = {p: calc_omega(ds, p) for p in [850, 700, 600, 500, 400]}
        omega1 = (omegas[850] + omegas[700]) / 2
        omega2 = (omegas[600] + omegas[500] + omegas[400]) / 3

    delta = omega2 - omega1

    shallow = xr.where(delta >= 0, delta, 0) * xr.where(-omega1 >= 0, 1, 0)
    deep = xr.where(-omega2 >= 0, -omega2, 0)
    D = xr.where(deep == 0, np.nan, shallow / deep)
    return D.mean(dim='time', skipna=True)


era5 = era5.sel(time='2020-06')
um = um.sel(time='2020-06')
icon = icon.sel(time='2020-06')
d_index_um = calc_d_index(um)
d_index_icon = calc_d_index(icon)
d_index_era5 = calc_d_index(era5)

##### Calculate diffs #####

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

lon = era5['lon'].values
lat = era5['lat'].values

# S index
this_nside_um = hp.get_nside(d_index_um) # probably the same for um and icon, but just being safe :)
this_nside_icon = hp.get_nside(d_index_icon)

cells_um = get_nn_lon_lat_index(this_nside_um, lon, lat)
cells_icon = get_nn_lon_lat_index(this_nside_icon, lon, lat)

s_um_era5_diff = s_index_um.isel(cell = cells_um) - s_index_era5
s_icon_era5_diff = s_index_icon.isel(cell = cells_icon) - s_index_era5

# projection = ccrs.Robinson(central_longitude=0)
# fig, axes = plt.subplots(figsize=(10, 6), subplot_kw={'projection': projection}, layout='constrained')
# axes.set_global()
# im = egh.healpix_show(s_index_icon,ax=axes)
# axes.coastlines()
# axes.set_title(f'UM S index')
# plt.savefig('figures/ICON_s_index.png')

# D index

this_nside_um = hp.get_nside(d_index_um) # probably the same for um and icon, but just being safe :)
this_nside_icon = hp.get_nside(d_index_icon)

cells_um = get_nn_lon_lat_index(this_nside_um, lon, lat)
cells_icon = get_nn_lon_lat_index(this_nside_icon, lon, lat)

d_um_era5_diff = d_index_um.isel(cell = cells_um) - d_index_era5
d_icon_era5_diff = d_index_icon.isel(cell = cells_icon) - d_index_era5

##### Plotting S index #####
nrow = 1
ncol = 3
fm_bottom = 1.5 / (4.4*nrow + 2)
fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

plt_colnames = [
    'ERA5',
    'UM-ERA5',
    'ICON-ERA5'
    ]
for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org=axs[jcol])
    axs[jcol].add_feature(cfeature.LAND,color='white',zorder=2,edgecolor=None,lw=0)
    axs[jcol].text(
        0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
        ha='left', va='bottom', transform=axs[jcol].transAxes)

# example to add colorbar
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=1, cm_interval1=0.05, cm_interval2=0.1, cmap='viridis',)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-0.5, cm_max=0.5, cm_interval1=0.05, cm_interval2=0.1, cmap='BrBG')

axs[0].pcolormesh(
        era5.lon, era5.lat,
        s_index_era5,
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())

axs[1].pcolormesh(
        s_um_era5_diff.lon, s_um_era5_diff.lat,
        s_um_era5_diff,
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())

axs[2].pcolormesh(
        s_icon_era5_diff.lon, s_icon_era5_diff.lat,
        s_icon_era5_diff,
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #plt_mesh, #
    ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks, extend='both',
    cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.04]))
cbar.ax.set_xlabel(f's index (unitless)')
cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #plt_mesh2, #
    ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks2, extend='both',
    cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.04]))
cbar2.ax.set_xlabel(f's index bias (unitless)')

fig.subplots_adjust(left=0.001, right=0.999, bottom=fm_bottom, top=0.94)
fig.savefig(F'figures/s_index_diffs.png')

##### Plotting D index #####
nrow = 1
ncol = 3
fm_bottom = 1.5 / (4.4*nrow + 2)
fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

plt_colnames = [
    'ERA5',
    'UM-ERA5',
    'ICON-ERA5'
    ]
for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org=axs[jcol])
    # axs[jcol].add_feature(cfeature.LAND,color='white',zorder=2,edgecolor=None,lw=0)
    axs[jcol].text(
        0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
        ha='left', va='bottom', transform=axs[jcol].transAxes)

# example to add colorbar
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=4, cm_interval1=0.25, cm_interval2=0.5, cmap='Blues_r',)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-4, cm_max=4, cm_interval1=0.5, cm_interval2=1, cmap='BrBG_r')

axs[0].pcolormesh(
        era5.lon, era5.lat,
        d_index_era5,
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())

axs[1].pcolormesh(
        d_um_era5_diff.lon, d_um_era5_diff.lat,
        d_um_era5_diff,
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())

axs[2].pcolormesh(
        d_icon_era5_diff.lon, d_icon_era5_diff.lat,
        d_icon_era5_diff,
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #plt_mesh, #
    ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks, extend='both',
    cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.04]))
cbar.ax.set_xlabel(f'D index (unitless)')
cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #plt_mesh2, #
    ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks2, extend='both',
    cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.04]))
cbar2.ax.set_xlabel(f'D index bias (unitless)')

fig.subplots_adjust(left=0.001, right=0.999, bottom=fm_bottom, top=0.94)
fig.savefig(F'figures/d_index_diffs.png')