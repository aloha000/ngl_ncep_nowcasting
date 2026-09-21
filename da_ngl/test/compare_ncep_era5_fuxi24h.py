

import pickle as pkl
import pandas as pd
import xarray as xr

obs_path = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/surf_ncep'
fuxi_zarr_path = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl/dataset/fuxi_europe_0p25_24h_70ch.zarr'
era5_zarr_path = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl/dataset/label_europe_0p25.zarr'

fuxi = xr.open_dataset(fuxi_zarr_path, engine='zarr')
era = xr.open_dataset(era5_zarr_path, engine='zarr')


with open('/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/surf_ncep/surf_2017123121.pkl','rb') as f:
    data = pkl.load(f)

