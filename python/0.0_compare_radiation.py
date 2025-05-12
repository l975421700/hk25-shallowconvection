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