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
# The function here calculate the mass distribution
# 
# 
############################################################

import os, sys
import pandas as pd
import numpy as np
from scipy.stats import norm

# Increase print precision and use scientific notation
np.set_printoptions(precision=3, suppress=False)


def run(particle_diameters_file_path, rho_char, mean_mass_distribution, std_mass_distribution):
    
    d_names = ["Low_d_(mm)", "High_d_(mm)"]

    # read particle diameters input file
    df_particle_sizes = pd.read_csv(particle_diameters_file_path)

    list_values = []
    # calculating mass distribution values
    for d_name in d_names:
        diameters_mm = np.array(df_particle_sizes[d_name], dtype=np.float64)
        particle_volume = (4/3)*3.14*(np.power(((diameters_mm/1000)/2), 3))
        particle_mass_mg = particle_volume*(rho_char*1000000)
        log10_particle_mass_mg = np.log10(particle_mass_mg)
        cdf_log10_particle_mass_mg = norm.cdf(log10_particle_mass_mg, loc=mean_mass_distribution, scale=std_mass_distribution)
        list_values.append(cdf_log10_particle_mass_mg)

    output = list_values[1] - list_values[0]

    return output