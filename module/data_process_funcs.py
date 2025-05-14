


modis_cmip6_var = {
    'Cloud_Mask_Fraction': 'clt',
    'Cloud_Mask_Fraction_Low': 'cll',
    'Cloud_Mask_Fraction_Mid': 'clm',
    'Cloud_Mask_Fraction_High': 'clh',
    'Cloud_Water_Path_Liquid': 'clwvi',
    'Cloud_Water_Path_Ice': 'clivi',
}


def read_MCD06COSP_M3(ifile, ivar2, ivar1):
    from netCDF4 import Dataset
    from datetime import datetime, timedelta
    import xarray as xr
    
    year = int(ifile[28:32])
    doy  = int(ifile[33:36])
    month = (datetime(year, 1, 1) + timedelta(days=doy - 1)).month
    
    ds = Dataset(ifile, mode='r')
    
    if ivar1 in ['clt', 'cll', 'clm', 'clh']:
        data = ds[ivar2]['Mean'][:].transpose()[None, :] * 100
    elif ivar1 in ['clwvi', 'clivi']:
        data = ds[ivar2]['Mean'][:].transpose()[None, :] / 1000
    
    ds_out = xr.DataArray(
        name=ivar1,
        data=data,
        dims=['time', 'lat', 'lon'],
        coords={
            'time': [datetime.strptime(f'{year}-{month:02d}-{1}', '%Y-%m-%d')],
            'lat': ds['latitude'][:],
            'lon': ds['longitude'][:] % 360
            }).sortby('lon')
    return(ds_out)


