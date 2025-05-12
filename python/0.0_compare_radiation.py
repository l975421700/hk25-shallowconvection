import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


ceres = xr.open_dataset('/g/data/er8/users/cd3022/hk25-ShallowConvection/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202502.nc')

print(ceres.variables)

ceres.toa_sw_all_mon.mean(dim='time').plot(
    cmap='plasma',
)
plt.savefig('toa_sw_flux_mean.png', dpi=300, bbox_inches='tight')

ger = xr.open_zarr('/g/data4/qx55/germany_node/d3hp003.zarr')

########################

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
sys.path.append('/home/548/cd3022/hk25-shallowconvection/module')
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
    cm_min=-400, cm_max=0, cm_interval1=0.1, cm_interval2=0.1, cmap='viridis',)
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