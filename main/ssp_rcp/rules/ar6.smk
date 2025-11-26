rule all:
    input:
        NETCDF_DIR + "/ar6_all.nc",
# -----------------------------
# Step 1: CSV → per-model NetCDF
# -----------------------------
rule merge_ar6_csv_to_netcdf:
    input:
        lambda wc: (
            glob.glob(f"{AR6_RAW}/{wc.var}/{wc.ssp}/{wc.model}/*.csv")
            if wc.ssp != "historical"
            else glob.glob(f"{AR6_RAW}/{wc.var}/historical/*.csv")
        )
    output:
        AR6_NETCDF + "/{var}/{ssp}_{model}.nc"
    run:
        import pandas as pd, numpy as np, xarray as xr

        if not input:
            print(f"No CSVs found for var={wildcards.var}, ssp={wildcards.ssp}, model={wildcards.model}")
            return

        all_da = []
        print(f"Processing CSVs for var={wildcards.var}, ssp={wildcards.ssp}, model={wildcards.model}")
        
        for f in input:
            df = pd.read_csv(f)
            if df.empty:
                continue

            # Extract time columns
            time_cols = [c for c in df.columns if c not in ["LON", "LAT"] and c.isdigit()]
            times = pd.to_datetime(time_cols, format="%Y%m%d", errors="coerce")
            if times.isna().all():
                print(f"Skipping {f}: no valid time columns")
                continue

            lon = np.sort(df["LON"].unique())
            lat = np.sort(df["LAT"].unique())
            data = np.full((len(times), len(lat), len(lon)), np.nan)

            for i, tcol in enumerate(time_cols):
                grid = df.pivot_table(values=tcol, index="LAT", columns="LON")
                grid = grid.reindex(index=lat, columns=lon)
                data[i] = grid.values

            da = xr.DataArray(
                data,
                dims=("time", "lat", "lon"),
                coords={"time": times, "lat": lat, "lon": lon},
                name=wildcards.var,
                attrs={"ssp": wildcards.ssp, "model": wildcards.model},
            )
            all_da.append(da)

        if not all_da:
            print(f"No valid data arrays for var={wildcards.var}, ssp={wildcards.ssp}, model={wildcards.model}")
            return

        ds = xr.concat(all_da, dim="time").sortby("time")
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        ds.to_netcdf(output[0])
        print(f"Saved per-model NetCDF: {output[0]}")

# -----------------------------
# Step 2: (Optional) Combine per-model NetCDF → per-variable NetCDF
# -----------------------------
rule combine_variable_netcdf:
    input:
        lambda wildcards: expand(
            AR6_NETCDF + "/{var}/{ssp}_{model}.nc",
            ssp=ssp_scenarios,
            model=ar6_variables,
            var=wildcards.var
        )
    output:
        AR6_NETCDF + "/merged/{var}.nc"
    run:
        import xarray as xr, os

        if not input:
            print(f"No NetCDF files to combine for var={wildcards.var}")
            return

        print(f"Combining {len(input)} files for var={wildcards.var}")
        ds_list = [xr.open_dataset(f) for f in input]
        combined = xr.concat(ds_list, dim="model")
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        combined.to_netcdf(output[0])
        print(f"Saved combined variable NetCDF: {output[0]}")


# -----------------------------
# Step 3: Merge all variables → ar6_all.nc
# -----------------------------
rule make_ar6_all_netcdf:
    input:
            expand(
                AR6_NETCDF + "/{var}.nc",
                var=ar6_variables
            )
    output:
        NETCDF_DIR + "/ar6_all.nc"
    run:
        import xarray as xr, os

        if not input:
            raise ValueError("No input NetCDF files found for merging!")

        print(f"Merging {len(input)} variable NetCDFs into final ar6_all.nc")
        ds_list = [xr.open_dataset(f) for f in input]
        combined = xr.concat(ds_list, dim="variable")
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        combined.to_netcdf(output[0])
        print(f"Saved final merged NetCDF: {output[0]}")