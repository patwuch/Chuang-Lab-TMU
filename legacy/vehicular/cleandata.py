import pyreadstat
import pandas as pd
import os
import glob
import cudf
import cupy as cp
import cuspatial

# === Load SPSS File with pyspssio ===
sav_path = "/home/patwuch/projects/vehicular/data/raw/Weather Stations-20250729T070752Z-1-001/Weather Stations/SAMPLE/2023.sav"
df, meta = pyreadstat.read_sav(sav_path)


# === Clean and Convert ===
df = df.drop(index=0).reset_index(drop=True)
df['GPS_X'] = pd.to_numeric(df['GPS_X'], errors='coerce')
df['GPS_Y'] = pd.to_numeric(df['GPS_Y'], errors='coerce')
df = df.dropna(subset=['GPS_X', 'GPS_Y'])

# === Load Weather Station CSVs ===
csv_dir = "/home/patwuch/projects/vehicular/data/raw/Weather Stations-20250729T070752Z-1-001/Weather Stations/測站_2023"
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
station_all = pd.concat([pd.read_csv(f, encoding='utf-8') for f in csv_files], ignore_index=True, sort=False)
station_all = station_all.drop(columns=['Unnamed: 27'], errors='ignore')

# === Load Station Location Data ===
location_file = "/home/patwuch/projects/vehicular/data/raw/Weather Stations-20250729T070752Z-1-001/Weather Stations/觀測站_Data.csv.xlsx"
location_df = pd.read_excel(location_file, usecols=["sitename", "twd97lon", "twd97lat", "siteid"])
location_df["siteid"] = location_df["siteid"].fillna(0).astype(str)
station_all = station_all.merge(location_df, left_on='測站', right_on='sitename', how='left')
station_all = station_all.dropna(subset=['twd97lon', 'twd97lat'])

# === Convert to cuDF ===
entries = cudf.DataFrame({
    "x": cp.asarray(df["GPS_X"]),
    "y": cp.asarray(df["GPS_Y"]),
})

stations = cudf.DataFrame({
    "x": cp.asarray(station_all["twd97lon"]),
    "y": cp.asarray(station_all["twd97lat"]),
})

# === Run cuSpatial knn search (k=1) ===
results = cuspatial.knn(stations, entries, k=1)

# === Map nearest station IDs back to df ===
nearest_indices = results["nearest_point_index"].to_numpy()
df["nearest_siteid"] = station_all.iloc[nearest_indices]["siteid"].values

# For each row in df, extract weather metrics from station_all and add as new columns
metrics = [
    'AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5',
    'RAINFALL', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED',
    'WS_HR', 'CH4', 'NMHC', 'THC'
]

# Convert date columns to datetime for accurate matching
df['OcDate'] = pd.to_datetime(df['OcDate'])
station_all['日期'] = pd.to_datetime(station_all['日期'])

# Melt station_all to long format: one row per siteid, date, hour, metric
station_long = station_all.melt(
    id_vars=['siteid', '日期', '測項'],
    value_vars=[str(h).zfill(2) for h in range(24)],
    var_name='Hour',
    value_name='value'
)

station_long = station_long[station_long['測項'].isin(metrics)]
station_long.set_index(['siteid', '日期', 'Hour', '測項'], inplace=True)

def get_weather_value(row, metric):
    try:
        siteid = str(row['nearest_siteid'])
        date = pd.to_datetime(row['OcDate'])
        hour = str(int(row['Hour'])).zfill(2)
        return station_long.loc[(siteid, date, hour, metric), 'value']
    except Exception:
        return None

for metric in metrics:
    df[f'weather_{metric}'] = df.apply(lambda row: get_weather_value(row, metric), axis=1)

df.to_csv("/home/patwuch/projects/vehicular/data/processed/Weather_Stations_2023.csv", index=False)