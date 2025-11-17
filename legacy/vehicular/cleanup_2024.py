# %%
import pandas as pd
import os
import glob
import cudf
import cupy as cp

csv_path = "/home/patwuch/projects/vehicular/data/raw/2024資料/2024.csv"
df = pd.read_csv(csv_path, encoding='Big5')
df


# %%
# === Clean and Convert ===
df = df.drop(index=0).reset_index(drop=True)
df['GPS_X'] = pd.to_numeric(df['GPS_X'], errors='coerce')
df['GPS_Y'] = pd.to_numeric(df['GPS_Y'], errors='coerce')
df = df.dropna(subset=['GPS_X', 'GPS_Y'])

# === Load Weather Station CSVs ===
station_dir = "/home/patwuch/projects/vehicular/data/raw/2024資料/全部天氣站_2024/全部天氣站_2024"
station_csvs = glob.glob(os.path.join(station_dir, '*.csv'))
station_all = pd.concat([pd.read_csv(f, encoding='utf-8') for f in station_csvs], ignore_index=True, sort=False)
station_all = station_all.drop(columns=['Unnamed: 27'], errors='ignore')

# %%
import re

location_file = "/home/patwuch/projects/vehicular/data/raw/2024資料/77個氣象站經緯度.xlsx"
location_df = pd.read_excel(location_file, usecols=["測站名稱", "經度", "緯度", "測站ID"])
import re
# Function: keep only Chinese characters
def extract_chinese(text):
    if pd.isna(text):
        return ""
    return "".join(re.findall(r'[\u4e00-\u9fff]+', str(text)))

# Clean Chinese-only names
location_df["chinese_name"] = location_df["測站名稱"].apply(extract_chinese)

# Sets for fast lookup
sitename_list = station_all["測站"].astype(str).tolist()

# Attempt substring-based matching
def fuzzy_match(name):
    for site in sitename_list:
        if site and site in name:  # site is contained in chinese_name
            return site
    return None

# Find fuzzy matches
location_df["fuzzy_match"] = location_df["chinese_name"].apply(fuzzy_match)

# Replace 測站名稱 with sitename where fuzzy_match succeeded
location_df.loc[location_df["fuzzy_match"].notna(), "測站名稱"] = \
    location_df.loc[location_df["fuzzy_match"].notna(), "fuzzy_match"]

# For reference, show what was changed
changed_rows = location_df.loc[location_df["fuzzy_match"].notna(), 
                               ["chinese_name", "fuzzy_match", "測站名稱"]]

print("Updated rows (Chinese name → matched sitename):")
print(changed_rows)


# %%
for col in location_df.select_dtypes(include=["object"]).columns:
    location_df[col] = (
        location_df[col]
        .astype(str)
        .str.replace(r"[\t\n\r\u3000]", "", regex=True)
        .str.strip()
    )


# %%
location_df = location_df.drop(columns=["chinese_name", "fuzzy_match"])

# %%
station_all = station_all.merge(location_df, left_on='測站', right_on='測站名稱', how='left')


# %%
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import re

# ---- 1. Convert DMS strings to decimal degrees ----
def maybe_convert_dms(val):
    if isinstance(val, str):
        # Remove any quotes or extra spaces
        val_clean = val.replace('"', '').replace("“", "").replace("”", "").strip()
        # Match DMS pattern: degrees° minutes' seconds
        match = re.match(r'(\d+)[°\s]+(\d+)[\'\s]+([\d.]+)', val_clean)
        if match:
            deg, minute, sec = map(float, match.groups())
            return deg + minute/60 + sec/3600
        else:
            # If it doesn't match DMS, try converting to float directly
            try:
                return float(val_clean)
            except:
                return np.nan
    # If it's already numeric, return as is
    return val

location_df['經度'] = location_df['經度'].apply(maybe_convert_dms)
location_df['緯度'] = location_df['緯度'].apply(maybe_convert_dms)

# ---- 2. Ensure df coordinates are numeric ----
df['GPS_X'] = pd.to_numeric(df['GPS_X'], errors='coerce')
df['GPS_Y'] = pd.to_numeric(df['GPS_Y'], errors='coerce')

# ---- 3. Separate rows with NaN coordinates ----
null_rows = df[df[['GPS_X', 'GPS_Y']].isna().any(axis=1)].copy()
valid_rows = df.dropna(subset=['GPS_X', 'GPS_Y']).copy()

# ---- 4. Drop NaN in location_df coordinates ----
location_clean = location_df.dropna(subset=['經度', '緯度']).copy()

# ---- 5. Build KDTree and find nearest station ----
station_coords = location_clean[['經度', '緯度']].values
tree = cKDTree(station_coords)

event_coords = valid_rows[['GPS_X', 'GPS_Y']].values
distances, indices = tree.query(event_coords, k=1)

# ---- 6. Assign nearest station info ----
valid_rows['nearest_site'] = location_clean.iloc[indices]['測站名稱'].values
valid_rows['site_id'] = location_clean.iloc[indices]['測站ID'].values

# ---- 7. Combine back if you want, keeping null_rows separate ----
df_processed = pd.concat([valid_rows, null_rows], ignore_index=True)


# %%
# For each row in df, extract weather metrics from station_all and add as new columns
metrics = [
    'AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5',
    'RAINFALL', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED',
    'WS_HR', 'CH4', 'NMHC', 'THC'
]

# Convert date columns to datetime for accurate matching
df_processed['OcDate'] = pd.to_datetime(df['OcDate'])
station_all['日期'] = pd.to_datetime(station_all['日期']).dt.date

# Create a dict of dicts:
# { metric: DataFrame indexed by (site_id, date), with columns 00–23 }
station_dict = {}
for metric in station_all['測項'].unique():
    subset = station_all[station_all['測項'] == metric].copy()
    subset = subset.set_index(['測站ID', '日期'])
    subset = subset[[str(h).zfill(2) for h in range(24)]]  # keep only 00–23 columns
    station_dict[metric] = subset

# === Ensure event times are aligned ===
df_processed['OcDate'] = pd.to_datetime(df_processed['OcDate']).dt.date
df_processed['Hour'] = df_processed['Hour'].astype(int).astype(str).str.zfill(2)

# === Per-metric lookup ===
for metric, table in station_dict.items():
    def lookup(row):
        key = (row['site_id'], row['OcDate'])
        if key in table.index:
            return table.loc[key, row['Hour']]
        return None

    df_processed[metric] = df_processed.apply(lookup, axis=1)


df_processed.to_csv("/home/patwuch/projects/vehicular/data/processed/2024_cleaned.csv", index=False, encoding='utf-8')


