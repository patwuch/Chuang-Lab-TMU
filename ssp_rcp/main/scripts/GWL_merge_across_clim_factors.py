import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path



ds_list = [xr.open_dataset(f) for f in input]
merged = xr.merge(ds_list)
output[0].parent.mkdir(parents=True, exist_ok=True)
merged.to_netcdf(output[0])
print("All processed!")

