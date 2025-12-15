import xarray as xr
import pandas as pd
ds = xr.open_dataset(input[0])

# Load config parameters
start_date = pd.to_datetime(config["start_date"])
end_date = pd.to_datetime(config["end_date"])
spatial_resolution = config["spatial_resolution"]
temporal_resolution = config["temporal_resolution"]
exclude_models = config["exclude_models"]

# Temporal slicing
ds_sliced = ds.sel(time=slice(start_date, end_date))

# Spatial resolution
if spatial_resolution == "1deg":
    ds_sliced = ds_sliced.coarsen(lat=2, lon=2, boundary="trim").mean()

# Temporal resolution
if temporal_resolution == "monthly":
    ds_sliced = ds_sliced.resample(time="1M").mean()

# Exclude models
if exclude_models:
    ds_sliced = ds_sliced.where(
        ~ds_sliced.model.isin(exclude_models),
        drop=True
    )

output[0].parent.mkdir(parents=True, exist_ok=True)
ds_sliced.to_netcdf(output[0])
