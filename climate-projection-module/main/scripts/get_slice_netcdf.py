import xarray as xr
import pandas as pd
from pathlib import Path

# Use Snakemake variables
ds = xr.open_dataset(snakemake.input[0])
print(ds.time.min().values, ds.time.max().values)
print(ds.time)
print(ds.time.dtype)
print(ds.time.attrs)

# Load config parameters
start_date = pd.to_datetime(snakemake.config["start_date"])
end_date = pd.to_datetime(snakemake.config["end_date"])
spatial_resolution = snakemake.config["spatial_resolution"]
temporal_resolution = snakemake.config["temporal_resolution"]
exclude_models = snakemake.config.get("exclude_models", [])

# Temporal slicing
ds_sliced = ds.sel(time=slice(start_date, end_date))

print(ds_sliced.time)
print(ds_sliced.time.dtype)
print(ds_sliced.time.min().values, ds_sliced.time.max().values)


# Spatial resolution
if spatial_resolution == "1deg":
    ds_sliced = ds_sliced.coarsen(lat=2, lon=2, boundary="trim").mean()
elif spatial_resolution == "pinpoint":
    ds_sliced = ds_sliced.mean(dim=["lat", "lon"], keep_attrs=True)

# Temporal resolution
if temporal_resolution == "monthly":
    ds_sliced = ds_sliced.resample(time="1M").mean()
# If daily, do nothing (already daily)
elif temporal_resolution == "daily":
    pass
else:
    raise ValueError(f"Unknown temporal_resolution: {temporal_resolution}")

# Exclude models
if exclude_models:
    ds_sliced = ds_sliced.where(
        ~ds_sliced.model.isin(exclude_models),
        drop=True
    )

# Save output
Path(snakemake.output[0]).parent.mkdir(parents=True, exist_ok=True)
ds_sliced.to_netcdf(snakemake.output[0])
print("Slicing done!")
