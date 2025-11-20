############################################################
#
# Developed by Majid Bavandpour
# Ph.D. Student @ UNR <majid.bavandpour@gmail.com>
# Summer 2025
#
############################################################
#
#
# The functions here are used to distribute particle in space and calculate the particel mass deposition.
#
#
############################################################

import numpy as np
import sys, os
from osgeo import gdal, osr
import shapely
import argparse
from pyproj import Transformer
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
import pandas as pd

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

import matplotlib.colors as mcolors

def create_white_to_brown_colormap(name='WhiteToBrown', num_colors=256):
    """
    Creates a custom Matplotlib colormap that transitions from white to a dark brown.
    
    Args:
        name (str): The name for the custom colormap.
        num_colors (int): The number of colors in the colormap (resolution).
        
    Returns:
        matplotlib.colors.LinearSegmentedColormap: The custom colormap object.
    """
    
    # Define the key colors for the ramp
    # Matplotlib expects colors as (R, G, B) tuples, where each value is 0-1.
    # We can also use named HTML colors or hex codes.
    
    # Option 1: Simple White to Dark Brown
    # colors = ['white', 'saddlebrown'] 
    
    # Option 2: More nuanced with a light tan/beige in between for smoother transition
    colors = ['#f0ecaa',      # White
            '#f0ecaa',      # Wheat (light tan/beige)
            '#a69165',
            "#664830"]      # Sienna (darker brown)
            # '#8B4513']    # SaddleBrown (even darker brown, if needed)

    # Create the colormap
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(name, colors, N=num_colors)
    
    return custom_cmap


def convert_latlon_epsg(lat, lon, target_epsg):
    """
    Convert coordinates from EPSG:4326 (lat, lon) to another EPSG.
    
    Args:
        lat (float): Latitude in degrees
        lon (float): Longitude in degrees
        target_epsg (int): EPSG code of target projection (e.g., 3857, 32633)

    Returns:
        (x, y): Coordinates in the target projection
    """
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def write_geotiff(array, geotransform, Xsize, Ysize, tif_file_path, crs=None, wkt=None):

    # reload(gdal)
    # reload(osr)

    driver = gdal.GetDriverByName("GTiff")
    
    array[array<=0] = np.nan

    if array.ndim > 2:
        dataset = driver.Create(tif_file_path, Xsize, Ysize, array.shape[0], gdal.GDT_Float32)
        # Upper Left x, Eeast-West px resolution, rotation, Upper Left y, rotation, North-South px resolution
        dataset.SetGeoTransform(geotransform)

        if not wkt:
            ref = osr.SpatialReference()
            ref.ImportFromEPSG(crs)
            dataset.SetProjection(ref.ExportToWkt())
        elif not crs:
            dataset.SetProjection(wkt)
        else:
            print("WARNING: no crs was set on the geotiff file...")

        for band in range(array.shape[0]):
            dataset.GetRasterBand(band + 1).WriteArray(array[band,:,:])
            dataset.GetRasterBand(band + 1).SetNoDataValue(-9999.0)
    else:
        dataset = driver.Create(tif_file_path, Xsize, Ysize, 1, gdal.GDT_Float32)

        # Upper Left x, Eeast-West px resolution, rotation, Upper Left y, rotation, North-South px resolution
        dataset.SetGeoTransform(geotransform)

        if not wkt:
            ref = osr.SpatialReference()
            ref.ImportFromEPSG(crs)
            dataset.SetProjection(ref.ExportToWkt())
        elif not crs:
            dataset.SetProjection(wkt)
        else:
            print("WARNING: no crs was set on the geotiff file...")
        
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.GetRasterBand(1).SetNoDataValue(-9999.0)
    
    dataset.FlushCache()
    del dataset


def extract(BBOX, xcell, ycell, polygon, FSProb):
    # BBOX = [minx, miny, maxx, maxy]
    # xcell = GeoTransform[1]  xCellSize
    # ycell = -GeoTransform[5] yCellSize


    ds = gdal.Rasterize('/vsimem/tmpfile', shapely.to_geojson(polygon), xRes=xcell, yRes=-ycell, allTouched=True,
                    outputBounds=BBOX, burnValues=FSProb, 
                    outputType=gdal.GDT_Float64)
    mask = ds.ReadAsArray()
    ds = None
    gdal.Unlink('/vsimem/tmpfile')
    # print(mask)
    # print(FSProb)

    mask[mask != FSProb] = 0

    return mask


def create_cone_polygon(apex, main_angle_deg, spread_angle_deg, inner_radius, outer_radius, num_points=100):
    """
    Create a cone polygon, using 0 degrees = North and positive clockwise.
    
    Parameters:
        apex: (x, y) tuple
        main_angle_deg: float, degrees (0=N, clockwise positive)
        spread_angle_deg: float, degrees
        inner_radius: float
        outer_radius: float
        num_points: int
        
    Returns:
        Shapely Polygon
    """
    # Convert to math angles
    main_angle = np.deg2rad(90 - main_angle_deg)
    spread_angle = np.deg2rad(spread_angle_deg)
    
    half_spread = spread_angle / 2

    start_angle = main_angle - half_spread
    end_angle = main_angle + half_spread

    outer_angles = np.linspace(start_angle, end_angle, num_points)
    inner_angles = np.linspace(end_angle, start_angle, num_points)

    outer_points = [
        (apex[0] + outer_radius * np.cos(theta),
         apex[1] + outer_radius * np.sin(theta))
        for theta in outer_angles
    ]

    inner_points = [
        (apex[0] + inner_radius * np.cos(theta),
         apex[1] + inner_radius * np.sin(theta))
        for theta in inner_angles
    ]

    points = outer_points + inner_points

    return Polygon(points)


def plot_polygons(outer_polygon, inner_polygon, outer_color='blue', inner_color='red', alpha=0.5):
    """
    Plot one polygon over another and return the Shapely Polygon object.
    
    Parameters:
        outer_polygon: Shapely Polygon representing the outer polygon.
        inner_polygon: Shapely Polygon representing the inner polygon.
        outer_color: Color of the outer polygon (default is blue).
        inner_color: Color of the inner polygon (default is red).
        alpha: Transparency of the inner polygon (default is 0.5).
    """
    # Unzip the points for the outer and inner polygons
    outer_x, outer_y = outer_polygon.exterior.xy
    inner_x, inner_y = inner_polygon.exterior.xy

    # Plot the outer polygon
    plt.fill(outer_x, outer_y, color=outer_color, alpha=0.5, label='Outer Polygon')
    plt.plot(outer_x, outer_y, color=outer_color)

    # Plot the inner polygon over the outer polygon
    plt.fill(inner_x, inner_y, color=inner_color, alpha=alpha, label='Inner Polygon')
    plt.plot(inner_x, inner_y, color=inner_color)

    # Set the aspect ratio to ensure the polygons are not distorted
    plt.gca().set_aspect('equal', adjustable='box')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()


def cone_arr_pdf(x, y, apex, main_angle_deg, spread_angle_deg, inner_radius, outer_radius, sigma_deg):
    """
    Compute the pdf value at (x, y) inside the cone.
    
    Parameters:
        x, y: array
        apex: (x0, y0)
        main_angle_deg: float, main direction (0=N, CW positive)
        spread_angle_deg: float, spread (total width)
        inner_radius: float
        outer_radius: float
        sigma_deg: float, std dev for angular spread (degrees)
        
    Returns:
        float: pdf value at (x,y) (0 if outside cone)
    """
    x0, y0 = apex

    
    # Relative coordinates
    dx = x - x0
    dy = y - y0

    
    # Polar coordinates
    r = np.hypot(dx, dy)
    theta_math = np.arctan2(dy, dx)  # math angle (0=East, CCW)
    

    # Convert to your system (0=North, CW positive)
    theta_deg = (90 - np.rad2deg(theta_math)) % 360

    
    # Center angle
    main_angle_deg = main_angle_deg % 360

    
    # Angular distance (carefully handle wraparound)
    delta_theta = ((theta_deg - main_angle_deg + 180) % 360) - 180  # Now between [-180, +180]

    
    # # Check if inside cone
    half_spread = spread_angle_deg / 2
    # if not (inner_radius <= r <= outer_radius):
    #     return 0.0
    # if not (-half_spread <= delta_theta <= half_spread):
    #     return 0.0

    # Now compute pdf
    sigma = sigma_deg
    p_theta = np.exp(-0.5 * (delta_theta / sigma)**2)

    p_theta[r<=inner_radius] = 0
    p_theta[r>=outer_radius] = 0

    p_theta[delta_theta<=-half_spread] = 0
    p_theta[delta_theta>=half_spread] = 0


    # Optional: normalization constant (depends on total integral)
    return p_theta


def cone_pdf(x, y, apex, main_angle_deg, spread_angle_deg, inner_radius, outer_radius, sigma_deg):
    """
    Compute the pdf value at (x, y) inside the cone.
    
    Parameters:
        x, y: coordinates
        apex: (x0, y0)
        main_angle_deg: float, main direction (0=N, CW positive)
        spread_angle_deg: float, spread (total width)
        inner_radius: float
        outer_radius: float
        sigma_deg: float, std dev for angular spread (degrees)
        
    Returns:
        float: pdf value at (x,y) (0 if outside cone)
    """
    x0, y0 = apex

    
    # Relative coordinates
    dx = x - x0
    dy = y - y0

    
    # Polar coordinates
    r = np.hypot(dx, dy)
    theta_math = np.arctan2(dy, dx)  # math angle (0=East, CCW)
    

    # Convert to your system (0=North, CW positive)
    theta_deg = (90 - np.rad2deg(theta_math)) % 360

    
    # Center angle
    main_angle_deg = main_angle_deg % 360

    
    # Angular distance (carefully handle wraparound)
    delta_theta = ((theta_deg - main_angle_deg + 180) % 360) - 180  # Now between [-180, +180]

    
    # Check if inside cone
    half_spread = spread_angle_deg / 2
    if not (inner_radius <= r <= outer_radius):
        return 0.0
    if not (-half_spread <= delta_theta <= half_spread):
        return 0.0

    # Now compute pdf
    sigma = sigma_deg
    p_theta = np.exp(-0.5 * (delta_theta / sigma)**2)


    # Optional: normalization constant (depends on total integral)
    return p_theta


def run(lat, lon, xcell, ycell, distance, norm_std, ash__wind_csv_file_path, mass_distance_data_folder, data_dir, smoke_plume_shps_folder_path = None, minDistance = 2, plot = False):

    sigma_deg = norm_std
    distance_m = distance*1000
    x, y = convert_latlon_epsg(lat, lon, 26910)
    BBOX = [x-(distance_m), y-distance_m, x+distance_m, y+distance_m]
    grid_size_x = int(2*distance_m/xcell)
    grid_size_y = int(2*distance_m/ycell)

    ash_wind_df = pd.read_csv(ash__wind_csv_file_path)
    dates = ash_wind_df["Date"]
    total_ash_kg = ash_wind_df["Ash_kg"]
    plume_apex_lats = ash_wind_df["apex_lat_deg"]
    plume_apex_lons = ash_wind_df["apex_lon_deg"]
    plume_main_dir = ash_wind_df["main_dir_deg"]
    plume_spread_deg = ash_wind_df["plume_spread_deg"]

    x_cords = np.linspace(x - distance_m, x + distance_m, grid_size_x)
    y_cords = np.linspace(y - distance_m, y + distance_m, grid_size_y)

    X, Y = np.meshgrid(x_cords, y_cords)


    outArr = np.zeros_like(X)
    geotransform = [x-distance_m, xcell, 0, y+distance_m, 0, -ycell]

    for i in range(len(dates)):

        print("\ndate: ", dates[i])
        dailyOut = np.zeros_like(X)
        tetst = dailyOut.copy()

        apex = (convert_latlon_epsg(plume_apex_lats[i], plume_apex_lons[i], 26910))
        main_angle = plume_main_dir[i] # * (2*np.pi/360)
        spread_angle = plume_spread_deg[i] # * (2*np.pi/360)

        mass_distance_df = pd.read_csv(os.path.join(mass_distance_data_folder, str(dates[i])+".csv"))
        mass_distribution = mass_distance_df["Mass_distribution"]
        inner_radius = mass_distance_df["Bin_short_distance_km"]
        outer_radius = mass_distance_df["Bin_far_distance_km"]

        for j in range(len(mass_distribution)):
            print("\rbin {}/{}                                                                      ".format(j, len(mass_distribution)), flush=True, end=" ")
            if inner_radius[j]>distance:
                print("\r a bin has been ignored due to long distance....", flush=True, end=" ")
                continue
            if inner_radius[j]<minDistance:
                print("this bin has been ignored due to low distance....", flush=True, end=" ")
                continue

            else:
                

                total_ash_in_bin_kg = mass_distribution[j]*total_ash_kg[i]
                ash_distribution_date_bin = np.zeros_like(X)

                # for row in range(grid_size_y):
                #     for col in range(grid_size_x):
                #         ash_distribution_date_bin[row, col] = cone_pdf(X[row, col], Y[row, col], apex, main_angle, spread_angle, inner_radius[j]*1000, outer_radius[j]*1000, sigma_deg)

                ash_distribution_date_bin = cone_arr_pdf(X, Y, apex, main_angle, spread_angle, inner_radius[j]*1000, outer_radius[j]*1000, sigma_deg)

                if np.sum(ash_distribution_date_bin) == 0:
                    print("\r a bin has been ignored due to zero sum....", flush=True, end=" ")
                    continue

                if np.isnan(np.sum(ash_distribution_date_bin)):
                    print("\r a bin has been ignored due to nan sum....", flush=True, end=" ")
                    continue

                ash_distribution_date_bin = ash_distribution_date_bin/np.sum(ash_distribution_date_bin)

                outArr = outArr + (ash_distribution_date_bin*total_ash_in_bin_kg)
                dailyOut = dailyOut + (ash_distribution_date_bin*total_ash_in_bin_kg)
                
                if plot:


                    # transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)

                    SMALL_SIZE = 7.5  # legend
                    MEDIUM_SIZE = 5   # rest

                    plt.rcParams['font.family'] = 'Calibri'

                    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
                    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
                    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
                    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
                    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
                    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

                    figuresFilder = os.path.join(data_dir, "Figures")
                    if not os.path.exists(figuresFilder):
                        os.mkdir(figuresFilder)

                    if os.path.exists(os.path.join(figuresFilder, "Fig_{}_{}.png".format(dates[i], j))):
                        continue

                    
                    cone_polygon = create_cone_polygon(apex, main_angle, spread_angle, minDistance*1000, distance*1000)
                    
                    min_data = 0
                    max_data = 10
                    gamma_value = 0.5
                    
                    fig, ax = plt.subplots(figsize=(2.5, 2.5))
                    fig.subplots_adjust(left=0.1, right=0.9, top=0.90, bottom=0.10)
                    tetst[tetst==0] = np.nan
                    cmapp = create_white_to_brown_colormap()
                    norm = mcolors.PowerNorm(gamma=gamma_value, vmin=min_data, vmax=max_data)

                    im = ax.pcolormesh(X, Y, tetst/10,  norm=norm,shading='auto', cmap=cmapp)
                    # im = ax.pcolormesh(X, Y, ash_distribution_date_bin*total_ash_in_bin_kg, shading='auto', cmap='viridis')

                    # dailyOut[dailyOut==0] = np.nan
                    # im = ax.pcolormesh(X, Y, dailyOut/10, norm=norm, shading='auto', cmap=cmapp)
                    cbar = fig.colorbar(im, ax=ax, shrink=0.35, pad=0.02)
                    cbar.set_label('Mass (g/m2)') #, fontsize=12)

                    n_data_ticks = np.linspace(0, 1, 2)
                    data_ticks = min_data + (max_data - min_data) * n_data_ticks**(1/gamma_value)
                    data_ticks = np.round(data_ticks, 0)

                    cbar.set_ticks(data_ticks)

                    cbar.ax.tick_params(pad=0, length=2, width=0.5)

                    # Reduce colorbar border width
                    for spine in cbar.ax.spines.values():
                        spine.set_linewidth(0.5)  # default is usually 1.0
                    
                    # Plot apex
                    ax.plot(apex[0], apex[1], 'ro', label='Apex', markersize=1)

                    x_ext, y_ext = cone_polygon.exterior.xy
                    ax.fill(x_ext, y_ext,facecolor='none', edgecolor='black', linewidth=0.5, label='Cone')


                    plt.ylim((4270000, 4330000))

                    # 2. Remove the frame by hiding the spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    # If you also want to remove the bottom and left axes:
                    # ax.spines['bottom'].set_visible(False)
                    # ax.spines['left'].set_visible(False)

                    # Optional: Remove the tick marks as well for a cleaner look
                    ax.tick_params(axis='both', which='both', length=0)
                    
                    # plt.title('Particle Mass Heatmap')
                    # plt.xlabel('X')
                    # plt.ylabel('Y')
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    plt.gca().set_aspect('equal')
                    plt.grid(False)
                    # plt.legend()
                    plt.savefig(os.path.join(figuresFilder, "Fig_{}_{}.png".format(dates[i], j)), dpi=1000)
                    
                    plt.clf()
                    plt.cla()
                    plt.close()
                    exit()


        # cone_polygon = create_cone_polygon(apex, main_angle, spread_angle, inner_radius, outer_radius)
        # polyTest = gpd.read_file(os.path.join(smoke_plume_shps_folder_path, str(dates[i])+".shp"))
        # polygon = polyTest.iloc[0]["geometry"]
        # plot_polygons(polygon, cone_polygon)

        write_geotiff(dailyOut[::-1], geotransform, dailyOut.shape[1], dailyOut.shape[0], os.path.join(data_dir, "particleMassMapDaily_{}.tif".format(dates[i])), crs=26910, wkt=None)

    # Upper Left x, Eeast-West px resolution, rotation, Upper Left y, rotation, North-South px resolution
    geotransform = [x-distance_m, xcell, 0, y+distance_m, 0, -ycell]
    write_geotiff(outArr[::-1], geotransform, outArr.shape[1], outArr.shape[0], os.path.join(data_dir, "particleMassMap.tif"), crs=26910, wkt=None)



