
rule AR6_convert_and_merge_across_years:
    input:
        ar6_csv_paths
    output:
        ar6_interim_netcdf_list
    params:
        grouping = ar6_groups
    conda:
        "/home/patwuch/Documents/projects/Chuang_Lab_TMU/main/ssp_rcp/envs/environment.yaml"
    log:
        "/home/patwuch/Documents/projects/Chuang_Lab_TMU/main/ssp_rcp/AR6_convert_and_merge_across_years.log"
    script:
        "/home/patwuch/Documents/projects/Chuang_Lab_TMU/main/ssp_rcp/scripts/AR6_convert_and_merge_across_years.py"

rule AR6_merge_across_models:
    input:
        ar6_interim_netcdf_list
    output:
        # One merged file per variable
        expand(AR6_NETCDF_DIR / "{factor}.nc", factor=ar6_clim_factors)
    params:
        all_vars = ar6_clim_factors,
        all_ssps = ssp_scenarios,
        AR6_NETCDF_DIR = AR6_NETCDF_DIR,
        AR6_INTERIM_DIR = AR6_INTERIM_DIR
    conda:
        "/home/patwuch/Documents/projects/Chuang_Lab_TMU/main/ssp_rcp/envs/environment.yaml"
    log:
        "/home/patwuch/Documents/projects/Chuang_Lab_TMU/main/ssp_rcp/AR6_merge_across_models.log"
    script:
        "/home/patwuch/Documents/projects/Chuang_Lab_TMU/main/ssp_rcp/scripts/AR6_merge_across_models.py"

# -------------------------------------------
# Step 3: Merge across all climate factors
# -------------------------------------------
rule AR6_merge_across_clim_factors:
    input:
        expand(
            AR6_NETCDF_DIR / "{factor}.nc",
            factor=ar6_clim_factors
        )
    output:
        NETCDF_DIR / "ar6_all.nc"
    conda:
        "/home/patwuch/Documents/projects/Chuang_Lab_TMU/main/ssp_rcp/envs/environment.yaml"
    script:
        "/home/patwuch/Documents/projects/Chuang_Lab_TMU/main/ssp_rcp/scripts/AR6_merge_across_clim_factors_netcdf.py"
