import pandas as pd
import matplotlib.pyplot as plt

csv_ang = r"C:\Users\mbavandpour\OneDrive - University of Nevada, Reno\PhD\FilesShouldShareWithBoxFolder\Facundo_Research\Code Repository\data\era5_wind_postprocessed_results_Facundo_day\2021-08-26_ang.csv"
csv_vel = r"C:\Users\mbavandpour\OneDrive - University of Nevada, Reno\PhD\FilesShouldShareWithBoxFolder\Facundo_Research\Code Repository\data\era5_wind_postprocessed_results_Facundo_day\2021-08-26_vel.csv"

ang_pd = pd.read_csv(csv_ang)
vel_pd = pd.read_csv(csv_vel)

ang = ang_pd["data"]
vel = vel_pd["data"]

plt.scatter(ang, vel)
plt.show()

