import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
ruleorder:
    GWL_convert_and_merge_across_years > GWL_merge_across_models

###############################################################################
# Helper discovery functions
###############################################################################

###############################################################################
# RULE: Merge CSVs → One NetCDF per (clim_factor, gwl)
###############################################################################

rule GWL_convert_and_merge_across_years:
    input:
        csvs=lambda w: sorted(
            (GWL_RAW_DIR / w.clim_factor / w.gwl).glob("*.csv"))
    output:
        GWL_NETCDF_DIR / "{clim_factor}" / "{gwl}.nc"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "GWL_convert_and_merge_across_years.py"
        
       

###############################################################################
# RULE: Combine all GWL NetCDFs for each climate factor
###############################################################################

rule GWL_merge_across_models:
    input:
        lambda w: sorted((GWL_NETCDF_DIR / w.clim_factor).glob("*.nc"))
    output:
        GWL_NETCDF_DIR / "combined_{clim_factor}.nc"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "GWL_merge_across_models.py"

###############################################################################
# RULE: Merge all climate factors into a single gwl_all.nc
###############################################################################

rule GWL_merge_across_climate_factors:
    input:
        lambda w: GWL_NETCDF_DIR / f"combined_{w.clim_factor}.nc"
    output:
        NETCDF_DIR / "{clim_factor}_all.nc"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "GWL_merge_across_climate_factors.py"


###############################################################################
# RULE: Slice GWL all.nc depending on provided configuration
###############################################################################
rule GWL_slice_gwl_all_netcdf:
    input:
        GWL_NETCDF_DIR / "{clim_factor}_all.nc"
    output:
        GWL_NETCDF_DIR / "{clim_factor}_sliced.nc"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "GWL_slice_gwl_all_netcdf.py" 
        

rule GWL_netcdf_slice_to_tsv:
    input:
        GWL_NETCDF_DIR / "{clim_factor}_sliced.nc"
    output:
        GWL_NETCDF_DIR / "{clim_factor}_sliced.tsv"
    conda:
        SSPRCP_MAIN_DIR / "envs/environment.yaml"
    script:
        SCRIPTS_DIR / "GWL_netcdf_slice_to_tsv.py"
        
