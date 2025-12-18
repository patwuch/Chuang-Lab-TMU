import xarray as xr
import numpy as np

# Load dataset
ds = xr.open_dataset("/home/patwuch/Documents/projects/Chuang_Lab_TMU/ssp_rcp/work/data/processed/NetCDF/TREAD_NetCDF/combined_rel_humid.nc")

# Print min and max date
if 'time' in ds.coords:
    if ds.time.size > 0:
        print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
    else:
        print("Time coordinate exists but is empty.")
else:
    print("No 'time' coordinate found in dataset.")

# Function to check masking
def check_fill_values(ds, fill_candidates=[-99.9, -9999]):
    for var in ds.data_vars:
        values = ds[var].values
        masked = np.isnan(values)
        n_masked = masked.sum()
        n_fill = np.isin(values, fill_candidates).sum()
        
        print(f"Variable: {var}")
        print(f"  Total elements: {values.size}")
        print(f"  Masked (NaN) count: {n_masked}")
        print(f"  Raw fill candidate count: {n_fill}")
        print("  ---")
        
check_fill_values(ds)
