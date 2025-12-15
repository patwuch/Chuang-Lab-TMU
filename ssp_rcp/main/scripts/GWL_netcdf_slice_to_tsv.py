import xarray as xr

ds = xr.open_dataset(input[0])

# Convert to tabular form
df = ds.to_dataframe().reset_index()

output[0].parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output[0], sep="\t", index=False)