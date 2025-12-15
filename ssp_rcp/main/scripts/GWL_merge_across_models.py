import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path



ds_list = []

for f in input:
    ds = xr.open_dataset(f)
    gwl_val = Path(f).stem
    for var in ds:
        ds[var] = ds[var].expand_dims({"gwl": [gwl_val]})
    ds_list.append(ds)

combined = xr.concat(ds_list, dim="gwl")
output[0].parent.mkdir(parents=True, exist_ok=True)
combined.to_netcdf(output[0])