from datetime import datetime
from osgeo import gdal, osr
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import os
import pandas as pd

def read_raster_subset(asc_path, bbox, plot=False):
  
    # --- Step 1: Read projection from .prj ---
    prj_path = os.path.splitext(asc_path)[0] + ".prj"
    if not os.path.exists(prj_path):
        raise FileNotFoundError(f"Projection file not found: {prj_path}")

    with open(prj_path, "r") as f:
        prj_wkt = f.read().strip()


    # --- Step 2: Open raster ---
    ds = gdal.Open(asc_path)

    arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
    gt = ds.GetGeoTransform()

    target_crs = pyproj.CRS.from_wkt(prj_wkt)
    transformer = pyproj.Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)

    x_origin, pixel_w, _, y_origin, _, pixel_h = gt
    n_cols = ds.RasterXSize
    n_rows = ds.RasterYSize

    minx = x_origin
    maxx = x_origin + pixel_w*n_cols

    miny = y_origin + pixel_h*n_rows
    maxy = y_origin

    min_lon, min_lat = transformer.transform(minx, miny)
    max_lon, max_lat = transformer.transform(maxx, maxy)

    lon_coords = np.linspace(min_lon, max_lon, n_cols)
    lat_coords = np.linspace(max_lat, min_lat, n_rows)

    lons, lats = np.meshgrid(lon_coords, lat_coords)

    # --- Step 5: Keep values within bbox ---
    mask = (lons >= bbox[0]) & (lons <= bbox[2]) & (lats >= bbox[1]) & (lats <= bbox[3])


    rows, cols = np.where(mask)
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()

    cropped_data = arr[rmin:rmax+1, cmin:cmax+1]

    # --- Step 6: Optional plot ---
    if plot:
        plt.figure(figsize=(7, 6))
        plt.imshow(cropped_data, cmap="rainbow")
        plt.colorbar(label="Value")
        plt.title("Raster with BBox Mask")
        plt.tight_layout()
        plt.show()

    return cropped_data



def list_files_by_pattern(directory, pattern):
    """
    List all files in a directory that end with a given pattern.

    Parameters
    ----------
    directory : str
        Path to the directory.
    pattern : str
        File ending pattern (e.g. '.asc', '_data.txt').

    Returns
    -------
    files : list of str
        Full paths of matching files.
    """
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(pattern)
    ]
    return files



import os
from datetime import datetime
import re

def extract_datetime_from_filename(file_path):
    """
    Extract a datetime object from a file name of the form:
    '..._MM-DD-YYYY_HHMM_...asc'

    Parameters
    ----------
    file_path : str
        Full or relative path to the file.

    Returns
    -------
    datetime.datetime
        Parsed datetime object.
    """
    filename = os.path.basename(file_path)
    
    # Regex pattern for MM-DD-YYYY_HHMM
    match = re.search(r'(\d{2}-\d{2}-\d{4})_(\d{4})', filename)
    if not match:
        raise ValueError(f"No date-time pattern found in filename: {filename}")
    
    date_str, time_str = match.groups()
    dt_str = f"{date_str}_{time_str}"
    
    # Parse to datetime object
    dt = datetime.strptime(dt_str, "%m-%d-%Y_%H%M")
    return dt



windninja_output_path = r"C:\Users\mbavandpour\OneDrive - University of Nevada, Reno\PhD\FilesShouldShareWithBoxFolder\Facundo_Research\Windninja_run\caldor_lcp\output_linux_runs\output"
out_path = r"C:\Users\mbavandpour\OneDrive - University of Nevada, Reno\PhD\FilesShouldShareWithBoxFolder\Facundo_Research\Code Repository\data\windninja_postprocessed_results_second_domain"
time_intervals = [[datetime(2021, 8, 25, 23, 0, 0, 0), datetime(2021, 8, 26, 23, 0, 0, 0)],
                  [datetime(2021, 8, 26, 23, 0, 1, 0), datetime(2021, 8, 27, 23, 0, 0, 0)],
                  [datetime(2021, 8, 27, 23, 0, 1, 0), datetime(2021, 8, 28, 23, 0, 0, 0)],
                  [datetime(2021, 8, 28, 23, 0, 1, 0), datetime(2021, 8, 29, 23, 0, 0, 0)],
                  [datetime(2021, 8, 29, 23, 0, 1, 0), datetime(2021, 8, 30, 23, 0, 0, 0)],
                  [datetime(2021, 8, 30, 23, 0, 1, 0), datetime(2021, 8, 31, 23, 0, 0, 0)],
                  [datetime(2021, 8, 31, 23, 0, 1, 0), datetime(2021, 9, 1, 23, 0, 0, 0)],
                  [datetime(2021, 9, 1, 23, 0, 1, 0), datetime(2021, 9, 2, 23, 0, 0, 0)],
                  [datetime(2021, 9, 2, 23, 0, 1, 0), datetime(2021, 9, 3, 23, 0, 0, 0)],
                  [datetime(2021, 9, 3, 23, 0, 1, 0), datetime(2021, 9, 4, 23, 0, 0, 0)],
                  [datetime(2021, 9, 4, 23, 0, 1, 0), datetime(2021, 9, 5, 23, 0, 0, 0)],
                  [datetime(2021, 9, 5, 23, 0, 1, 0), datetime(2021, 9, 7, 23, 0, 0, 0)],
                  [datetime(2021, 9, 7, 23, 0, 1, 0), datetime(2021, 9, 10, 23, 0, 0, 0)]
                  ]
bbox = (-120.307030, 38.536967, -119.917810, 38.962548) # first domain (-120.6752, 38.55, -119.9152, 39.202)



velocity_files = list_files_by_pattern(windninja_output_path, "_vel.asc")
direction_files = list_files_by_pattern(windninja_output_path, "_ang.asc")

data_vel = {}
data_ang = {}

for i in range(len(velocity_files)):

    velocity_file = velocity_files[i]
    direction_file = direction_files[i]
    date_time = extract_datetime_from_filename(velocity_file)

    print(date_time.strftime("%Y-%m-%d %H:%M"))

    vel_arr = read_raster_subset(velocity_file, bbox, plot=False)
    ang_arr = read_raster_subset(direction_file, bbox, plot=False)

    for time_interval in time_intervals:
        if (date_time>time_interval[0]) and (date_time<time_interval[1]):
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
    pd_ang["data"] = list((data_ang[time_interval[1].strftime("%Y-%m-%d")]*data_vel[time_interval[1].strftime("%Y-%m-%d")])/np.max(data_vel[time_interval[1].strftime("%Y-%m-%d")]))
    pd_ang.to_csv(os.path.join(out_path, time_interval[1].strftime("%Y-%m-%d")+"_ang.csv"))