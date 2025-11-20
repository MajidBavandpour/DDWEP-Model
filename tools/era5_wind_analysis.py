from datetime import datetime
from osgeo import gdal, osr
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import os
import pandas as pd
import os
from datetime import datetime
import re
import xarray as xr
import datetime
from zoneinfo import ZoneInfo
import pytz
 

era5_wind_data_path = r"C:\Users\mbavandpour\OneDrive - University of Nevada, Reno\PhD\FilesShouldShareWithBoxFolder\Facundo_Research\Windninja_run\ERA5\5by5_dayhours"
out_path = r"C:\Users\mbavandpour\OneDrive - University of Nevada, Reno\PhD\FilesShouldShareWithBoxFolder\Facundo_Research\Code Repository\data\era5_wind_postprocessing_5by5_dayhours"
time_intervals = [[datetime.datetime(2021, 8, 25, 23, 0, 0, 0), datetime.datetime(2021, 8, 26, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 8, 26, 23, 0, 1, 0), datetime.datetime(2021, 8, 27, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 8, 27, 23, 0, 1, 0), datetime.datetime(2021, 8, 28, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 8, 28, 23, 0, 1, 0), datetime.datetime(2021, 8, 29, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 8, 29, 23, 0, 1, 0), datetime.datetime(2021, 8, 30, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 8, 30, 23, 0, 1, 0), datetime.datetime(2021, 8, 31, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 8, 31, 23, 0, 1, 0), datetime.datetime(2021, 9, 1, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 9, 1, 23, 0, 1, 0), datetime.datetime(2021, 9, 2, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 9, 2, 23, 0, 1, 0), datetime.datetime(2021, 9, 3, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 9, 3, 23, 0, 1, 0), datetime.datetime(2021, 9, 4, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 9, 4, 23, 0, 1, 0), datetime.datetime(2021, 9, 5, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 9, 5, 23, 0, 1, 0), datetime.datetime(2021, 9, 7, 23, 0, 0, 0)],
                  [datetime.datetime(2021, 9, 7, 23, 0, 1, 0), datetime.datetime(2021, 9, 10, 23, 0, 0, 0)]
                  ]

data_vel = {}
data_ang = {}

pst = pytz.timezone('US/Pacific')

nc_files = os.listdir(era5_wind_data_path)
for file in nc_files:
    ds = xr.open_dataset(os.path.join(era5_wind_data_path, file))

    time_values_utc = ds["valid_time"].values

    valid_time = [
        datetime.datetime.fromtimestamp(
            t.astype('datetime64[s]').astype(int),
            tz=datetime.timezone.utc
        )
        for t in time_values_utc
        ]
    
    valid_time_pst = [t.astimezone(ZoneInfo("America/Los_Angeles")) for t in valid_time]

    for i, time in enumerate(time_values_utc):

        print(valid_time_pst[i])

        u10 = ds["u10"].sel(valid_time = time).values
        v10 = ds["v10"].sel(valid_time = time).values

        vel_arr = np.sqrt(u10**2 + v10**2)
        ang_arr = (np.degrees(np.arctan2(u10, v10)) + 360) % 360
        ang_arr = (ang_arr + 180) % 360

        for time_interval in time_intervals:
            if (valid_time_pst[i]>pst.localize(time_interval[0])) and (valid_time_pst[i]<pst.localize(time_interval[1])):
                if time_interval[1].strftime("%Y-%m-%d") in list(data_vel.keys()):
                    data_vel[time_interval[1].strftime("%Y-%m-%d")] = np.concatenate((data_vel[time_interval[1].strftime("%Y-%m-%d")], vel_arr.ravel()))
                    data_ang[time_interval[1].strftime("%Y-%m-%d")] = np.concatenate(( data_ang[time_interval[1].strftime("%Y-%m-%d")], ang_arr.ravel()))
                else:
                    data_vel[time_interval[1].strftime("%Y-%m-%d")] = vel_arr.ravel()
                    data_ang[time_interval[1].strftime("%Y-%m-%d")] = ang_arr.ravel()

        print("Done!")
        


print("writing the results files ....")
for time_interval in time_intervals:
    print(time_interval[1].strftime("%Y-%m-%d"))

    print("writing wind velocity data ...")
    pd_vel = pd.DataFrame()
    pd_vel["data"] = list(data_vel[time_interval[1].strftime("%Y-%m-%d")])
    pd_vel.to_csv(os.path.join(out_path, time_interval[1].strftime("%Y-%m-%d")+"_vel.csv"))

    print("writing wind direction data ...")
    pd_ang = pd.DataFrame()
    pd_ang["data"] = list((data_ang[time_interval[1].strftime("%Y-%m-%d")])) #*data_vel[time_interval[1].strftime("%Y-%m-%d")])/np.max(data_vel[time_interval[1].strftime("%Y-%m-%d")]))
    pd_ang.to_csv(os.path.join(out_path, time_interval[1].strftime("%Y-%m-%d")+"_ang.csv"))