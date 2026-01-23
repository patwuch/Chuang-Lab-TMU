# %%
from datetime import datetime, timedelta
import os

from ipyfilechooser import FileChooser
import ipywidgets as widgets
import pandas as pd
import geemap
import ee

from setting.utils import SelectFilesButton, monthlist, date_format_concersion, read_shp_date_data, read_bands_statics, make_temp_file, cbind_chirps
from setting.config import settings

# %%
# After executing this line of code for the first use, you can get the authentication number linked to Google.
Map = geemap.Map()
# Authenticate the Google earth engine with google account
ee.Initialize() 

# %%
my_button = SelectFilesButton()
my_button

# %%
file_name, start, end = read_shp_date_data(my_button)
widgets.HBox([file_name, start, end])

# %%
band_name, statics = read_bands_statics(settings.chrisp_bands_list)
widgets.HBox([band_name, statics])

# %%
# give the output floder and flie name
folder_name = make_temp_file('data_all_google_earth_engine_chirps')
folder = FileChooser()
display(folder)

# %%
time_list = monthlist(start.value, end.value)
states = geemap.shp_to_ee("".join(my_button.files))

chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filter(ee.Filter.date(datetime.strptime(start.value.strftime("%Y-%m-%d"), "%Y-%m-%d"), datetime.strptime(end.value.strftime("%Y-%m-%d"),"%Y-%m-%d")+timedelta(days=1))) \
        .map(lambda image: image.select(band_name.value)) \
        .map(lambda image: image.clip(states)) \
        .map(lambda image: image.reproject(crs=settings.crs))

chirps.toBands()
out_dir = os.path.expanduser(folder_name)
out_dem_stats = os.path.join(out_dir, 'chirps_{}.csv'.format(statics.value))

if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
geemap.zonal_statistics(chirps, states, out_dem_stats, statistics_type=statics.value, scale=1000)

data_temp = pd.read_csv(out_dem_stats)

data = []

column_name_list = data_temp.columns.tolist()
c = []
d = []

for k in zip(column_name_list[:]):
        c.append(k[0][0])
        d.append(k[0])

        data = []
for j in range(0, len(column_name_list), len(band_name.value)):

        date_str = data_temp.columns[j][:8]

        # 检查日期格式并提取数据
        if all(m.isdigit() for m in c[j:j+len(band_name.value)]) == True:
                
                # 提取数据
                df = data_temp.loc[:, d[j:j+len(band_name.value)]]

                df[file_name.value] = data_temp.loc[:, [file_name.value]]
                                
                # 创建新的日期和DOY列
                df.insert(0, 'Date', '')
                df['Date'] = date_format_concersion(date_str, output_format='%Y/%m/%d')

                df.insert(1, 'Doy', '')
                df['Doy'] = datetime.strptime(date_str, '%Y%m%d').strftime('%j')
                
                # 重命名列
                colnames = ['Date', 'Doy']
                colnames.extend(list(band_name.value))
                colnames.append(file_name.value)
                df.columns = [colnames]
                
                data.append(df)
        else:
                continue

appended_data = pd.concat(data, axis=0, ignore_index=True)

appended_data.to_csv(out_dem_stats, index=False) #Output the file with date and doy back

# %%
cbind_chirps(statics.value, out_dir, band_name.value, folder)

# %%



