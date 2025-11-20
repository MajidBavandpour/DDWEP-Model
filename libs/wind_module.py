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
# This module use wind data from wind staions to calculate mean daily wind direction, mean daily wind speed, and daily standard deviation of the wind direction
# 
# 
############################################################


import numpy as np
import os, sys
import pandas as pd
from datetime import datetime


def run(data_directory, station_names, station_weights, time_intervals):
    '''
        station_names   :: a list contains wind station names like ["OWNC", "KTLV"]
        station_weights :: a list contains weights for stations like [0.75, 0.25]
        time_intervals  :: a list contains list of start and end date object of each interval for daily it would be start and end datetime object of a day

        this function return main_dir_deg, plume_spread_deg, and wind_spd
    '''

    # intialize variables
    main_dir_deg = np.zeros(len(time_intervals))
    plume_spread_deg = np.zeros(len(time_intervals))
    wind_spd = np.zeros(len(time_intervals))

    # number of stations
    num_stations = len(station_names)

    # iteration over stations to read data df of each station into a list
    stations_df_list = []
    for station in station_names:
        df_station = pd.read_csv(os.path.join(data_directory, "inputs", "wind_station_{}.csv".format(station)))
        stations_df_list.append(df_station)

    # iteration over each row of data for each station to change datetime data to a datetime object
    for station_index in range(num_stations):
        df_station = stations_df_list[station_index]
        datetime_obj_list = []
        for row_index in range(len(df_station)):
            datetime_obj_list.append(datetime.strptime(df_station.iloc[row_index]["Date_Time"], "%m/%d/%Y %H:%M"))

        stations_df_list[station_index]["datetime_obj"] = datetime_obj_list


    # iteration over stations to calculate mean wind speed, wind direction, and std of wind direction in intervals
    stations_mean_wind_speed = []
    stations_mean_wind_direction = []
    stations_std_wind_direction = []

    for station_index in range(num_stations):

        station_mean_wind_speed = []
        station_mean_wind_direction = []
        station_std_wind_direction = []

        df_station = stations_df_list[station_index]

        # iteration over each interval to calculate mean and std
        for time_interval_index in range(len(time_intervals)):

            # filters data in the time intervals
            start_time = time_intervals[time_interval_index][0]
            end_time = time_intervals[time_interval_index][1]
            temp_df = df_station[df_station["datetime_obj"]>start_time]
            temp_df = temp_df[temp_df["datetime_obj"]<end_time]

            wind_spd_arr = np.array(temp_df["wnd_sp"].tolist())
            wind_dir_arr = np.array(temp_df["wnd_dir"].tolist())

            mean_wind_spd = float(np.mean(temp_df["wnd_sp"]))
            mean_wind_dir = float(np.sum(wind_spd_arr*wind_dir_arr)/np.sum(wind_spd_arr))
            std_wind_dir = float(np.sqrt(np.sum(wind_spd_arr*np.power((wind_dir_arr-mean_wind_dir), 2))/np.sum(wind_spd_arr)))

            station_mean_wind_speed.append(mean_wind_spd)
            station_mean_wind_direction.append(mean_wind_dir)
            station_std_wind_direction.append(std_wind_dir)

        stations_mean_wind_speed.append(np.array(station_mean_wind_speed))
        stations_mean_wind_direction.append(np.array(station_mean_wind_direction))
        stations_std_wind_direction.append(np.array(station_std_wind_direction))


    # combine stations data
    for station_index in range(num_stations):
        main_dir_deg = main_dir_deg + stations_mean_wind_direction[station_index]*station_weights[station_index]
        plume_spread_deg = plume_spread_deg + stations_std_wind_direction[station_index]*station_weights[station_index]
        wind_spd = wind_spd + stations_mean_wind_speed[station_index]*station_weights[station_index]

    main_dir_deg = main_dir_deg - 180

    return main_dir_deg, plume_spread_deg, wind_spd

