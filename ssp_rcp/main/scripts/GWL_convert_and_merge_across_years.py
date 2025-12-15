import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

all_da = []

for f in input:
    df = pd.read_csv(f)
    if df.empty:
        raise ValueError(f"Empty CSV: {f}")

    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns in {f}")

    time_cols = [c for c in df.columns if c not in ["LON", "LAT"] and c.isdigit()]
    times = pd.to_datetime(time_cols, format="%Y%m%d")

    lon = np.sort(df["LON"].unique())
    lat = np.sort(df["LAT"].unique())

    data = np.full((len(times), len(lat), len(lon)), np.nan)

    for i, tcol in enumerate(time_cols):
        grid = df.pivot_table(values=tcol, index="LAT", columns="LON")
        grid = grid.reindex(index=lat, columns=lon)
        data[i] = grid.values

    parts = f.name.split("_")
    ssp = next((p for p in parts if p.startswith("ssp")), "unknown")
    model = parts[parts.index(ssp) + 1] if ssp in parts else "unknown"
    year = parts[-1].split(".")[0]

    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat, "lon": lon},
        name=wildcards.clim_factor,
        attrs={
            "ssp": ssp,
            "model": model,
            "gwl": wildcards.gwl,
            "year": year,
        },
    )
    all_da.append(da)

ds = xr.concat(all_da, dim="time").sortby("time").groupby("time").first()
output[0].parent.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(output[0])