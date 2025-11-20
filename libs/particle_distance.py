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
# The function here calculates distance of travel for each particle size 
# 
# 
############################################################

import os, sys
import pandas as pd
import numpy as np
import libs.terminal_velocity as terminal_velocity

def run(time_intervals, particle_diameters_file_path, q_dot_kw, wind_spd, rho_air, rho_char, T_zero, constant_drag_coefficient, g, mu, cp, max_lofting_height, drag_coefficient_constant_method):

    d_names = ["Low_d_(mm)", "High_d_(mm)"]
    
    # creating date keys
    datetime_keys = []
    for datetime_index in range(len(time_intervals)):
        datetime_keys.append(time_intervals[datetime_index][1].strftime("%Y%m%d"))
    
    # read particle diameters input file
    df_particle_sizes = pd.read_csv(particle_diameters_file_path)

    # calculating terminal velocity
    df_terminal_velocity = terminal_velocity.calc(particle_diameters_file_path, rho_air, rho_char, drag_coefficient_constant_method, constant_drag_coefficient, g, mu)

    # iterate over time intervals to create distance for each bin of particle diameter
    particle_distances = {}
    for datetime_index, datetime_key in enumerate(datetime_keys):

        print("calculating distances for: ", datetime_key)

        wind_velocity_ref = wind_spd[datetime_index]
        q_dot_ref = q_dot_kw[datetime_index]

        # iteration over low and high particle sizes
        distances = []
        for d_name in d_names:
            diameters_m = np.array(df_particle_sizes[d_name])/1000
            terminal_velocity_ref = np.array(df_terminal_velocity["vt_{}".format(d_name)])
            drag_coefficient_ref = np.array(df_terminal_velocity["dr_{}".format(d_name)])

            # calculating lofting heights
            # lofting_heights = np.power(((q_dot_ref)/(rho_air*cp*T_zero*np.sqrt(g))), (2/5))*np.power((40*constant_drag_coefficient*rho_air/(4*diameters_m*rho_char*g)), (3/2))
            lofting_heights = np.power(((q_dot_ref)/(rho_air*cp*T_zero*np.sqrt(g))), (2/5))*np.power((40*drag_coefficient_ref*rho_air/(4*diameters_m*rho_char*g)), (3/2))
            lofting_heights[lofting_heights>max_lofting_height] = max_lofting_height

            # calculating wind profile
            wind_velocity_profile = wind_velocity_ref*(np.power((lofting_heights/10), 0.25))

            # calculating distances
            distance_km = (lofting_heights*wind_velocity_profile/terminal_velocity_ref)/1000

            distances.append(distance_km)

        particle_distances[datetime_key] = distances




    return datetime_keys, particle_distances

