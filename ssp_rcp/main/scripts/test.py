from pathlib import Path

GWL_RAW_DIR = Path("/home/patwuch/Documents/projects/Chuang_Lab_TMU/ssp_rcp/work/data/raw/GWL")
clim_factor = "prec"
gwl = "GWL1.5"

files = sorted((GWL_RAW_DIR / clim_factor / gwl).glob("*.csv"))
print(files)