############################################################
# Developed by Majid Bavandpour(a) and Facundo Scordo(b)
# (a) Ph.D. Student @ UNR <majid.bavandpour@gmail.com>
# (b) Research Faculty @ UNR <fscordo@unr.edu>
# Summer 2025
#
############################################################

############################################################
# 
# 
# This is the main file to run the model.
# 
# 
############################################################



import os, sys
import pandas as pd
import numpy as np
from datetime import datetime
import libs.wind_module as wind
import libs.total_ash_sensible_heat as ash_heat
import libs.particle_distance as particle_distance
import libs.particle_mass_dist as particle_mass_dist
import libs.particle_spatial_distribution as particle_spatial_distribution

#### inputs ####

## framework general inputs
run_name = "allday_rChar_150_ash_7_Dc_0.68"
rho_char = 150
ash_precent = 0.07
drag_coefficient_constant_method = True
mean_mass_distribution = -2.4

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
plume_apex_lats = [38.631, 38.62, 38.612, 38.61, 38.65, 38.652, 38.74, 38.749, 38.749, 38.749, 38.749, 38.742, 38.742]
plume_apex_lons = [-120.35, -120.348, -120.34, -120.331, -120.236, -120.165, -120.14, -120.141, -120.14, -120.141, -120.141, -120.139, -120.133]
data_directory = r"data"
rho_air = 1.293                       # Air density in kg/m3
# rho_char = 200                       # Char density in kg/m3
T_zero = 293                          # Temperature of the ambient air in K
constant_drag_coefficient = 0.68
g = 9.81                              # Gravitational constant  in m/s2
mu = 1.81e-5                          # Air dynamic viscosity in Pa
cp = 1                                # Specific heat capacity of air at constant pressure in KJ/kg/K

## wind processing inputs
wind_processing = True
wind_data_type = "era5"    # "stations" or "era5"
era5_wind_data_path = r"C:data\inputs\era5_wind_data_allday.csv"
wind_station_names = ["OWNC", "KTLV"]
wind_station_weights = [0.75, 0.25]

## ash and heat processing inputs
ash_heat_processing = True
kmz_path = r"data\inputs\ir_raw_data\KMZ Files\Polygons"
sb40_landfire = r"data\inputs\landfire_maps\LF2020_FBFM40_200_CONUS\LC20_F40_200.tif"
CBD_path = r"data\inputs\landfire_maps\LF2020_CBD_200_CONUS\LC20_CBD_200.tif"
CH_path = r"data\inputs\landfire_maps\LF2020_CH_200_CONUS\LC20_CH_200.tif"
CBH_path = r"data\inputs\landfire_maps\LF2020_CBH_200_CONUS\LC20_CBH_200.tif"
# ash_precent = 0.05

## particle distance processing inputs
particle_distance_processing = True
particle_diameters_file_path = r"data\inputs\particle_diameters.csv"
max_lofting_height = 10000 # meters
sf_fl_factor = 1    # factor to increase the surface fuel load
cp_fl_factor = 1     # factor to increase the canopy fuel load
# drag_coefficient_constant_method = True

## particle mass distribution processing inputs
particle_mass_dist_processing = True
# mean_mass_distribution = -2.25
std_mass_distribution = 0.55

## particle spatial distribution model inputs
lat = 38.631                         # coordinate of the center of the domain
lon = -120.35                        # coordinate of the center of the domain
xcell = 100                          # meters
ycell = 100                          # meters
min_allowable_distance = 1           # km
max_allowable_distance = 110         # km
norm_std = 10                        # deg
plot = False                         # if true you need to specify smoke_plume_shps_folder_path
smoke_plume_shps_folder_path = None

#### inputs ####



#### processing ####
## creating data directory
data_directory_case = os.path.join(data_directory, run_name)
if not os.path.exists(data_directory_case):
    os.mkdir(data_directory_case)

## wind data processing
if wind_processing:
    if wind_data_type == "stations":
        main_dir_deg, plume_spread_deg, wind_spd = wind.run(data_directory, wind_station_names, wind_station_weights, time_intervals)
    if wind_data_type == "era5":
        
        if not os.path.exists(era5_wind_data_path):
            sys.exit("{} file does not exist!!!". format(era5_wind_data_path))

        era5_wind_data_df = pd.read_csv(era5_wind_data_path)
        main_dir_deg = np.array(era5_wind_data_df["main_dir_deg"])
        plume_spread_deg = np.array(era5_wind_data_df["plume_spread_deg"])
        wind_spd = np.array(era5_wind_data_df["wind_spd"])

    else:
        sys.exit("wrong wind_data_type!!!")

## ash and heat processing
if ash_heat_processing:
    total_ash, q_dot_kw = ash_heat.run(kmz_path, data_directory_case, sb40_landfire, CBD_path, CH_path, CBH_path, ash_precent, sf_fl_factor, cp_fl_factor)

## particle distances processing
if particle_distance_processing:
    datetime_keys, particle_distances = particle_distance.run(time_intervals, particle_diameters_file_path, q_dot_kw, wind_spd, rho_air, rho_char, T_zero, constant_drag_coefficient, g, mu, cp, max_lofting_height, drag_coefficient_constant_method)

## particle mass distribution
if particle_mass_dist_processing:
    particle_mass_dist_values = particle_mass_dist.run(particle_diameters_file_path, rho_char, mean_mass_distribution, std_mass_distribution)

## writing particle spatial function inputs
ash_wind_data_df = pd.DataFrame()
ash_wind_data_df["Date"] = datetime_keys
ash_wind_data_df["Ash_kg"] = list(total_ash)
ash_wind_data_df["apex_lat_deg"] = plume_apex_lats
ash_wind_data_df["apex_lon_deg"] = plume_apex_lons
ash_wind_data_df["main_dir_deg"] = list(main_dir_deg)
ash_wind_data_df["plume_spread_deg"] = list(plume_spread_deg)

ash_wind_data_csv_path = os.path.join(data_directory_case, "ash_wind_data.csv")
ash_wind_data_df.to_csv(ash_wind_data_csv_path)

mass_distance_data_directory = os.path.join(data_directory_case, "mass_distance_data")
if not os.path.exists(mass_distance_data_directory):
    os.mkdir(mass_distance_data_directory)

for datetime_key in datetime_keys:
    Bin_far_distance_km = particle_distances[datetime_key][0]
    Bin_short_distance_km = particle_distances[datetime_key][1]

    df_to_write = pd.DataFrame()
    df_to_write["Mass_distribution"] = list(particle_mass_dist_values)
    df_to_write["Bin_far_distance_km"] = list(Bin_far_distance_km)
    df_to_write["Bin_short_distance_km"] = list(Bin_short_distance_km)

    df_to_write.to_csv(os.path.join(mass_distance_data_directory, (datetime_key+".csv")))

## run particle statial distribution model
if not os.path.exists(os.path.join(data_directory_case, "particle_raster_outputs")):
    os.mkdir(os.path.join(data_directory_case, "particle_raster_outputs"))
    
if smoke_plume_shps_folder_path and os.path.exists(smoke_plume_shps_folder_path):
    particle_spatial_distribution.run(lat, lon, xcell, ycell, max_allowable_distance, norm_std, ash_wind_data_csv_path, mass_distance_data_directory, os.path.join(data_directory_case, "particle_raster_outputs"), smoke_plume_shps_folder_path, min_allowable_distance, plot)
else:
    particle_spatial_distribution.run(lat, lon, xcell, ycell, max_allowable_distance, norm_std, ash_wind_data_csv_path, mass_distance_data_directory, os.path.join(data_directory_case, "particle_raster_outputs"), min_allowable_distance, plot=plot)

print("Process is completed!!!")
#### processing ####

