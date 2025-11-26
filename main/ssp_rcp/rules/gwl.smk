# ----------------------------------------
# Branch: GWL
# ----------------------------------------

rule merge_gwl_csvs:
    input:
        lambda wildcards: glob.glob(f"{GWL_DIR}/{wildcards.var}/{wildcards.gwl}/*.csv")
    output:
        GWL_NETCDF + "/{var}/{gwl}.nc"
    run:
        all_da = []

        for f in input:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                raise ValueError(f"Failed to read CSV {f}: {e}")

            if df.empty:
                raise ValueError(f"CSV file {f} is empty!")

            # Duplicate column check
            dup_cols = df.columns[df.columns.duplicated()]
            if len(dup_cols) > 0:
                raise ValueError(f"Duplicate columns in CSV {f}: {list(dup_cols)}")

            # Extract time columns (YYYYMMDD)
            time_cols = [c for c in df.columns if c not in ["LON", "LAT"] and c.isdigit()]

            try:
                times = pd.to_datetime(time_cols, format="%Y%m%d", errors="coerce")
            except Exception as e:
                raise ValueError(f"Error parsing dates in {f}: {e}")

            # Check duplicate timestamps
            if len(time_cols) != len(set(time_cols)):
                raise ValueError(f"Duplicate time columns detected in {f}: {time_cols}")

            # Sort lon/lat
            lon = np.sort(df["LON"].unique())
            lat = np.sort(df["LAT"].unique())

            nlon, nlat = len(lon), len(lat)
            data = np.full((len(times), nlat, nlon), np.nan)

            # Convert each time column to a 2D grid
            for i, tcol in enumerate(time_cols):
                grid = df.pivot_table(values=tcol, index="LAT", columns="LON")
                grid = grid.reindex(index=lat, columns=lon)
                data[i] = grid.values

            # Parse metadata from filename
            fname = os.path.basename(f)
            parts = fname.split("_")
            ssp = next((p for p in parts if p.startswith("ssp")), "unknown")

            try:
                model = parts[parts.index(ssp) + 1]
            except:
                model = "unknown"

            da = xr.DataArray(
                data,
                dims=("time", "lat", "lon"),
                coords={"time": times, "lat": lat, "lon": lon},
                name=wildcards.var,
                attrs={
                    "ssp": ssp,
                    "model": model,
                    "gwl": wildcards.gwl,
                    "year": parts[-1].split(".")[0],
                },
            )
            all_da.append(da)

        if not all_da:
            raise ValueError(f"No CSVs found for {wildcards.var}, {wildcards.gwl}")

        # Concatenate and clean timestamps
        ds = xr.concat(all_da, dim="time")
        ds = ds.sortby("time")
        ds = ds.groupby("time").first()

        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        ds.to_netcdf(output[0])


rule combine_gwl_netcdf:
    input:
        lambda wc: expand(
            GWL_NETCDF + "/{var}/{gwl}.nc",
            var=wc.var,
            gwl=gwls
        )
    output:
        GWL_NETCDF + "/combined_{var}.nc"
    run:
        ds_list = [xr.open_dataset(f) for f in input]

        for ds, f in zip(ds_list, input):
            gwl_val = os.path.basename(f).replace(".nc", "")
            for varname in ds.data_vars:
                ds[varname] = ds[varname].expand_dims({"gwl": [gwl_val]})

        combined = xr.concat(ds_list, dim="gwl")
        combined.to_netcdf(output[0])


rule make_gwl_all_netcdf:
    input:
        expand(GWL_NETCDF + "/combined_{var}.nc", var=ar6_variables)
    output:
        NETCDF_DIR + "/gwl_all.nc"
    run:
        ds_list = [xr.open_dataset(f) for f in input]
        merged = xr.merge(ds_list)
        merged.to_netcdf(output[0])
        print("All processed!")

