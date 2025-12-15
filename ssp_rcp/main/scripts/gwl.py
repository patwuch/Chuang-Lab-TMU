import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import xarray as xr

# -------------------------
# Configuration
# -------------------------
GWL_RAW_DIR = Path("/home/patwuch/projects/chuang_lab_tmu/work/ssp_rcp/data/raw/GWL")
GWL_NETCDF_DIR = Path("/home/patwuch/projects/chuang_lab_tmu/work/ssp_rcp/data/processed/NetCDF/GWL")
NETCDF_DIR = Path("/home/patwuch/projects/chuang_lab_tmu/work/ssp_rcp/data/processed/NetCDF")

# -------------------------
# Helper discovery functions
# -------------------------
def find_clim_factors():
    """Return all climate factor directories inside the raw GWL folder."""
    return [p.name for p in GWL_RAW_DIR.iterdir() if p.is_dir()]

def find_gwls(clim_factor):
    """Return all GWL-level directories inside a given climate factor folder."""
    folder = GWL_RAW_DIR / clim_factor
    return [p.name for p in folder.iterdir() if p.is_dir()]

clim_factors = find_clim_factors()
print("Climate factors:", clim_factors)
for cf in clim_factors:
    print(cf, find_gwls(cf))
# -------------------------
# Step 1: Merge CSV → per-(clim_factor, GWL) NetCDF
# -------------------------
for cf in clim_factors:
    gwls = find_gwls(cf)
    for gwl in gwls:
        csv_files = list((GWL_RAW_DIR / cf / gwl).glob("*.csv"))
        if not csv_files:
            continue

        print(f"Processing CSVs for {cf}/{gwl}")
        all_da = []

        for f in csv_files:
            df = pd.read_csv(f)
            if df.empty:
                print(f"Skipping empty CSV: {f}")
                continue

            # Check for duplicate columns
            dup = df.columns[df.columns.duplicated()]
            if len(dup) > 0:
                raise ValueError(f"Duplicate columns in {f}: {list(dup)}")

            # Identify numeric columns that may be dates
            time_cols = [c for c in df.columns if c not in ["LON", "LAT"] and c.isdigit()]
            valid_time_cols = []
            times_list = []

            # Parse dates, skip invalid ones
            for t in time_cols:
                try:
                    ts = pd.to_datetime(t, format="%Y%m%d")
                    valid_time_cols.append(t)
                    times_list.append(ts)
                except ValueError:
                    print(f"Skipping invalid date column: {t}")

            if not valid_time_cols:
                print(f"No valid date columns in {f}, skipping file.")
                continue

            times = pd.DatetimeIndex(times_list)

            lon = np.sort(df["LON"].unique())
            lat = np.sort(df["LAT"].unique())
            data = np.full((len(times), len(lat), len(lon)), np.nan)

            # Fill data array
            for i, tcol in enumerate(valid_time_cols):
                grid = df.pivot_table(values=tcol, index="LAT", columns="LON")
                grid = grid.reindex(index=lat, columns=lon)
                data[i] = grid.values

            # Extract metadata
            fname = f.name
            parts = fname.split("_")
            ssp = next((p for p in parts if p.startswith("ssp")), "unknown")
            model = parts[parts.index(ssp) + 1] if ssp in parts else "unknown"
            year = parts[-1].split(".")[0]

            da = xr.DataArray(
                data,
                dims=("time", "lat", "lon"),
                coords={"time": times, "lat": lat, "lon": lon},
                name=cf,
                attrs={"ssp": ssp, "model": model, "gwl": gwl, "year": year},
            )
            all_da.append(da)

        if not all_da:
            print(f"No valid data arrays for {cf}/{gwl}, skipping.")
            continue

        # Concatenate all CSVs for this (cf, gwl)
        ds = xr.concat(all_da, dim="time").sortby("time").groupby("time").first()
        out_path = GWL_NETCDF_DIR / cf / f"{gwl}.nc"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_path)
        print(f"Saved per-GWL NetCDF: {out_path}")


# -------------------------
# Step 2: Combine per-GWL → per-clim_factor NetCDF
# -------------------------
for cf in clim_factors:
    files = sorted((GWL_NETCDF_DIR / cf).glob("*.nc"))
    if not files:
        continue

    ds_list = []
    for f in files:
        ds = xr.open_dataset(f)
        gwl_val = f.stem
        for var in ds:
            ds[var] = ds[var].expand_dims({"gwl": [gwl_val]})
        ds_list.append(ds)

    combined = xr.concat(ds_list, dim="gwl")
    out_path = GWL_NETCDF_DIR / f"combined_{cf}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(out_path)
    print(f"Saved combined NetCDF for climate factor {cf}: {out_path}")

# -------------------------
# Step 3: Merge all climate factors → gwl_all.nc
# -------------------------
combined_files = [GWL_NETCDF_DIR / f"combined_{cf}.nc" for cf in clim_factors]
ds_list = [xr.open_dataset(f) for f in combined_files if f.exists()]

merged = xr.merge(ds_list)
out_path = NETCDF_DIR / "gwl_all.nc"
out_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_netcdf(out_path)
print(f"Saved final merged NetCDF: {out_path}")
print("All processed!")
