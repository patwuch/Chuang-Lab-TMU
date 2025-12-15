import xarray as xr
import pandas as pd
import numpy as np
import random
from pathlib import Path


# --- Snakemake variables ---
inputs = list(snakemake.input)         # all CSV files
outputs = list(snakemake.output)       # all desired NetCDF outputs
groups = snakemake.params.grouping     # dict: { "CMCC-ESM2.nc": [csv1, csv2, ...], ... }
# --- Functions ---
def load_csv_as_dataset(csv_path):
    """Convert a gridded CSV (LON, LAT, YYYYMMDD...) into an xarray Dataset."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Warning: Error reading {csv_path}: {e}", flush=True)
        return None

    if df.empty:
        print(f"Warning: {csv_path} is empty, skipping", flush=True)
        return None

    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    if "LON" not in df.columns or "LAT" not in df.columns:
        print(f"Warning: {csv_path} missing LON/LAT columns, skipping", flush=True)
        return None

    lon = df["LON"].values
    lat = df["LAT"].values

    time_cols = [c for c in df.columns if c not in ("LON", "LAT")]
    time = pd.to_datetime(time_cols, errors="coerce")

    if time.isna().all():
        print(f"Warning: {csv_path} has invalid dates, skipping", flush=True)
        return None

    data = df[time_cols].values
    data = np.where(data == -99.9, np.nan, data)

    ds = xr.Dataset(
        {"value": (("point", "time"), data)},
        coords={"lon": ("point", lon), "lat": ("point", lat), "time": time}
    )
    return ds

# --- Main processing loop ---
for i, (nc_name, csv_list) in enumerate(groups.items()):
    print(f"Processing {nc_name} with {len(csv_list)} CSV files...", flush=True)

    # Load datasets and force eager loading to avoid dask threads
    datasets = [ds.load() for ds in (load_csv_as_dataset(csv) for csv in csv_list) if ds is not None]
    if not datasets:
        print(f"Warning: No valid datasets for {nc_name}, skipping...", flush=True)
        continue

    combined = xr.concat(datasets, dim="time").sortby("time")

    # Save using Snakemake-defined output
    out_path = Path(outputs[i])
    combined.to_netcdf(out_path, engine="netcdf4")
    print(f"Saved {out_path}", flush=True)

    # --- Close all datasets to release file handles ---
    combined.close()
    for ds in datasets:
        ds.close()

    # Free memory
    del combined
    del datasets
    import gc
    gc.collect()

print("Finished all merges into .nc!", flush=True)

# Force dask scheduler back to synchronous, just in case
import dask
dask.config.set(scheduler='synchronous')

import os
os._exit(0)