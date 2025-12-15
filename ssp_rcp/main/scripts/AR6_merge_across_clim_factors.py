import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np
import glob
import os

# -------------------------
# Step 3: Merge all variables → final NetCDF
# -------------------------
files = snakemake.input
out_path = snakemake.output

if not files:
    raise ValueError("No variable NetCDF files found for merging!")

ds_list = []

for f in files:
    ds = xr.open_dataset(f)
    var_name = list(ds.data_vars)[0]
    ds = ds.expand_dims({"variable": [var_name]})
    ds_list.append(ds)

combined_all = xr.concat(ds_list, dim="variable", combine_attrs="override")
combined_all = combined_all.fillna(float("nan"))


os.makedirs(out_path.parent, exist_ok=True)
combined_all.to_netcdf(out_path)
print(f"Saved final merged NetCDF: {out_path}")
