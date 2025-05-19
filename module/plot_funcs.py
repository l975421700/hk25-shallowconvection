

# region get_nn_lon_lat_index

def get_nn_lon_lat_index(nside, lons, lats):
    import xarray as xr
    import numpy as np
    import healpy as hp
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        hp.ang2pix(nside, lons2, lats2, nest=True, lonlat=True),
        coords=[("lat", lats), ("lon", lons)],
    )


# endregion


# region funcs for global map plot

def ticks_labels(xmin, xmax, ymin, ymax, xspacing, yspacing):
    import numpy as np
    
    # get the x ticks
    xticks_pos = np.arange(xmin, xmax + 1e-4, xspacing)
    if not isinstance(xspacing, int):
        xticks_pos = np.around(xticks_pos, 1)
    else:
        xticks_pos = xticks_pos.astype('int')
    
    # Associate with '° W', '°', and '° E'
    xticks_label = [''] * len(xticks_pos)
    for i in np.arange(len(xticks_pos)):
        if (xticks_pos[i] > 180):
            xticks_pos[i] = xticks_pos[i] - 360
        
        if (abs(xticks_pos[i]) == 180) | (xticks_pos[i] == 0):
            xticks_label[i] = str(abs(xticks_pos[i])) + '°'
        elif xticks_pos[i] < 0:
            xticks_label[i] = str(abs(xticks_pos[i])) + '° W'
        elif xticks_pos[i] > 0:
            xticks_label[i] = str(xticks_pos[i]) + '° E'
    
    # get the y ticks
    yticks_pos = np.arange(ymin, ymax + 1e-4, yspacing)
    if not isinstance(yspacing, int):
        yticks_pos = np.around(yticks_pos, 1)
    else:
        yticks_pos = yticks_pos.astype('int')
    
    # Associate with '° N', '°', and '° S'
    yticks_label = [''] * len(yticks_pos)
    for i in np.arange(len(yticks_pos)):
        if yticks_pos[i] < 0:
            yticks_label[i] = str(abs(yticks_pos[i])) + '° S'
        if yticks_pos[i] == 0:
            yticks_label[i] = str(yticks_pos[i]) + '°'
        if yticks_pos[i] > 0:
            yticks_label[i] = str(yticks_pos[i]) + '° N'
    
    return xticks_pos, xticks_label, yticks_pos, yticks_label


import numpy as np
import cartopy.crs as ccrs
def globe_plot(
    ax_org=None,
    figsize=np.array([8.8, 4.4]) / 2.54,
    projections = ccrs.Mollweide(central_longitude=180),
    add_atlas=True, atlas_color='black', lw=0.1,
    add_grid=True, grid_color='gray',
    add_grid_labels=False, ticklabel = None, labelsize=10,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.01, fm_top=0.99
    ):
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    
    if (ticklabel is None):
        ticklabel = ticks_labels(0, 360, -90, 90, 60, 30)
    
    if (ax_org is None):
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, subplot_kw={'projection': projections},)
    else:
        ax = ax_org
    
    if add_grid_labels:
        ax.set_xticks(ticklabel[0],)
        ax.set_xticklabels(ticklabel[1], fontsize=labelsize)
        ax.set_yticks(ticklabel[2],)
        ax.set_yticklabels(ticklabel[3], fontsize=labelsize)
        ax.tick_params(length=1, width=lw * 2)
    
    if add_atlas:
        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '10m', edgecolor=atlas_color,
            facecolor='none', lw=lw)
        ax.add_feature(coastline, zorder=2)
        borders = cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land', '10m',
            edgecolor=atlas_color, facecolor='none', lw=lw)
        ax.add_feature(borders, zorder=2)
    
    if add_grid:
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), linewidth=lw * 0.75, zorder=2,
            color=grid_color, linestyle='--')
        gl.xlocator = mticker.FixedLocator(ticklabel[0])
        gl.ylocator = mticker.FixedLocator(ticklabel[2])
    
    if (ax_org is None):
        fig.subplots_adjust(
            left=fm_left, right=fm_right, bottom=fm_bottom, top=fm_top)
    
    if (ax_org is None):
        return fig, ax
    else:
        return ax


from matplotlib.ticker import FuncFormatter
@FuncFormatter
def remove_trailing_zero_pos(x, pos):
    return ('%f' % x).rstrip('0').rstrip('.')


def plt_mesh_pars(
    cm_min, cm_max, cm_interval1, cm_interval2, cmap,
    clip=True, reversed=True, asymmetric=False,
    ):
    '''
    #-------- Input
    
    #-------- Output
    
    '''
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib import cm
    import matplotlib.pyplot as plt
    
    pltlevel = np.arange(cm_min, cm_max + 1e-4, cm_interval1, dtype=np.float64)
    pltticks = np.arange(cm_min, cm_max + 1e-4, cm_interval2, dtype=np.float64)
    pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=clip)
    pltcmp = plt.get_cmap(cmap, len(pltlevel)-1)
    
    if(reversed):
        pltcmp = pltcmp.reversed()
    
    if (asymmetric):
        cm_range = np.max((abs(cm_min), abs(cm_max))) * 2
        pltcmp = plt.get_cmap(cmap, int(cm_range / cm_interval1))
        
        if(reversed):
            pltcmp = pltcmp.reversed()
        
        if (abs(cm_min) > abs(cm_max)):
            pltcmp = ListedColormap(
                [pltcmp(i) for i in range(pltcmp.N)][:(len(pltlevel)-1)])
        else:
            pltcmp = ListedColormap(
                [pltcmp(i) for i in range(pltcmp.N)][-(len(pltlevel)-1):])
    
    return([pltlevel, pltticks, pltnorm, pltcmp])


'''
# The following is an example code to plot multiple globe plots.

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

nrow = 1
ncol = 4
fm_bottom = 1.5 / (4.4*nrow + 2)
fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

plt_colnames = ['Dataset', 'ERA5', 'UM', 'ICON']
for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org=axs[jcol])
    axs[jcol].text(
        0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
        ha='left', va='bottom', transform=axs[jcol].transAxes)

# example to add colorbar
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.1, cmap='viridis',)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.2, cmap='BrBG')

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #plt_mesh, #
    ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks, extend='both',
    cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.04]))
cbar.ax.set_xlabel('Colorbar label 1')
cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #plt_mesh2, #
    ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks2, extend='both',
    cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.04]))
cbar2.ax.set_xlabel('Colorbar label 2')

fig.subplots_adjust(left=0.001, right=0.999, bottom=fm_bottom, top=0.94)
fig.savefig('figures/test.png')

'''
# endregion

