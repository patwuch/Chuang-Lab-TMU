print("Starting...")
import os
from pathlib import Path
from collections import defaultdict
import xarray as xr

# =============================================================================
# Snakemake inputs/outputs
# =============================================================================
all_inputs = snakemake.input          # list of all input files (interim per-model NetCDFs)
all_outputs = snakemake.output        # list of output files (one merged file per variable)

AR6_NETCDF_DIR = Path(snakemake.params.AR6_NETCDF_DIR)
AR6_INTERIM_DIR = Path(snakemake.params.AR6_INTERIM_DIR)
all_vars = snakemake.params.all_vars  # e.g., ar6_clim_factors
all_ssps = snakemake.params.all_ssps  # optional, for ordering

# =============================================================================
# Cleaning function
# =============================================================================
def clean_ds(ds):
    """Remove duplicate time stamps and sort."""
    if "time" not in ds:
        return ds

    t = ds.time.to_index()

    if t.has_duplicates:
        print("  -> Removing duplicate timestamps")
        ds = ds.sel(time=~t.duplicated())

    ds = ds.sortby("time")
    return ds

# =============================================================================
# STEP 1 — CLEAN ALL PER-MODEL NETCDF FILES
# =============================================================================
print("\n================ CLEANING PER-MODEL NETCDF FILES ================\n")

for var in all_vars:
    var_dir = AR6_INTERIM_DIR / var
    if not var_dir.exists():
        print(f"Variable directory {var_dir} does not exist, skipping")
        continue

    files = sorted(var_dir.glob("*/*.nc"))
    if not files:
        print(f"No files found for variable {var}")
        continue

    for f in files:
        print(f"CLEANING: {f}")
        with xr.open_dataset(f) as ds:
            ds_clean = clean_ds(ds)
            ds_clean.load()
        ds_clean.to_netcdf(f)
        print(f"  -> cleaned and saved")

# =============================================================================
# STEP 2 — COMBINE MODELS PER VARIABLE
# =============================================================================
print("\n===== COMBINING MODELS INTO PER-VARIABLE NETCDF =====\n")

for i, var in enumerate(all_vars):
    print(f"\nProcessing variable: {var}")

    var_dir = AR6_INTERIM_DIR / var
    files = sorted(var_dir.glob("*/*.nc"))
    if not files:
        print(f"No files found for variable {var}, skipping")
        continue

    # Group datasets by SSP
    ssp_groups = defaultdict(list)
    for f in files:
        with xr.open_dataset(f, chunks={"time": 10}) as ds:
            ds = clean_ds(ds)
            ssp = ds.attrs.get("ssp", "historical")
            model = ds.attrs.get("model", "unknown_model")
            ds_expanded = ds.expand_dims({"ssp": [ssp], "model": [model]})
            ssp_groups[ssp].append(ds_expanded)

    # Concat models within each SSP
    ssp_datasets = []
    for ssp, group in ssp_groups.items():
        if len(group) == 0:
            continue
        print(f"  + Combining {len(group)} models for SSP {ssp}")
        ds_ssp = xr.concat(group, dim="model", combine_attrs="override", coords="minimal")
        ssp_datasets.append(ds_ssp)

    if len(ssp_datasets) == 0:
        print(f"No valid SSP datasets for variable {var}, skipping")
        continue

    # Concat across SSPs
    combined = xr.concat(ssp_datasets, dim="ssp", combine_attrs="override", coords="minimal")
    combined = combined.sortby("time")

    # Save final combined file to Snakemake-defined output
    out_path = Path(all_outputs[i])
    combined.to_netcdf(out_path)
    print(f"✔ Saved combined variable file → {out_path}")
