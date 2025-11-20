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
# This module use IR data and LANDFIRE fuel maps to calculate total ash and heat release.
# 
# 
############################################################




import geopandas as gpd
from osgeo import gdal, osr, ogr
import shapely
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
from matplotlib.collections import PatchCollection
import libs.ir_processing as ir_processing
import libs.fuel_load_map as fuel_load_map



def plot_vector_over_raster(raster_ds: gdal.Dataset, vector_ds: ogr.DataSource):
    # Convert raster GDAL dataset to rasterio dataset for plotting
    with rasterio.open(raster_ds.GetDescription()) as src:
        fig, ax = plt.subplots(figsize=(10, 10))
        rasterio.plot.show(src, ax=ax)

        # Get raster geotransform
        gt = raster_ds.GetGeoTransform()

        # Get vector layer
        layer = vector_ds.GetLayer()
        patches = []

        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue

            # Handle multipolygon and polygon
            if geom.GetGeometryType() == ogr.wkbPolygon:
                polygons = [geom]
            elif geom.GetGeometryType() == ogr.wkbMultiPolygon:
                polygons = [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())]
            else:
                continue  # Skip other geometry types

            for poly in polygons:
                ring = poly.GetGeometryRef(0)  # exterior ring
                if ring is None:
                    continue
                points = ring.GetPoints()
                # Convert geo-coordinates to pixel coordinates
                pixels = [(int((x - gt[0]) / gt[1]), int((y - gt[3]) / gt[5])) for x, y in points]
                patch = MplPolygon(pixels, closed=True, edgecolor='red', facecolor='none', linewidth=1)
                patches.append(patch)

        p = PatchCollection(patches, match_original=True)
        ax.add_collection(p)

        plt.title("Vector over Raster")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.grid(False)
        plt.show()




def extract(BBOX, xcell, ycell, polygon, FSProb):

    # Create the GeoJSON string from the polygon
    geojson_str = shapely.to_geojson(polygon)
    mem_file = '/vsimem/tmpfile.geojson'
    gdal.FileFromMemBuffer(mem_file, geojson_str.encode('utf-8'))

    # Open the in-memory GeoJSON as a vector datasource
    vector_ds = gdal.OpenEx(mem_file, gdal.OF_VECTOR)

    # Create the raster dataset
    raster_filename = '/vsimem/tmpfile.tif'
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(raster_filename, 
                       int((BBOX[2] - BBOX[0]) / xcell), 
                       int((BBOX[3] - BBOX[1]) / ycell), 
                       1, 
                       gdal.GDT_Float64)

    # Set geotransform and projection
    ds.SetGeoTransform((BBOX[0], xcell, 0, BBOX[3], 0, -ycell))
    # ds.SetProjection('EPSG:26910')  # Assuming WGS84, change if needed
    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(4326)
    # ds.SetProjection(srs.ExportToWkt())

    # Rasterize the vector data
    ret = gdal.Rasterize(ds, vector_ds, burnValues=[FSProb], allTouched=True)
    if ret != 1:
        print("Rasterization failed")
        return None

    # plot_vector_over_raster(ds, vector_ds)

    # Extract the mask
    mask = ds.ReadAsArray()
    ds = None
    gdal.Unlink(raster_filename)
    gdal.Unlink(mem_file)

    # Apply the FSProb mask
    mask[mask != FSProb] = 0

    

    # === Added plotting part starts here ===
    
    # Plot mask and polygon overlay
    # fig, ax = plt.subplots(figsize=(8, 8))

    # # Show the mask
    # ax.imshow(mask, cmap='gray', origin='upper', extent=(BBOX[0], BBOX[2], BBOX[1], BBOX[3]))

    # # Overlay polygon(s) with semi-transparent fill
    # if polygon.is_empty:
    #     ax.text(0.5, 0.5, "Empty Geometry", ha='center', va='center', transform=ax.transAxes)
    # elif isinstance(polygon, Polygon):
    #     coords_2d = [(x, y) for x, y, *rest in polygon.exterior.coords]
    #     patch = MplPolygon(coords_2d, closed=True,
    #                     edgecolor='blue', facecolor='none', alpha=0.6)
    #     ax.add_patch(patch)
    # elif isinstance(polygon, MultiPolygon):
    #     for poly in polygon.geoms:
    #         coords_2d = [(x, y) for x, y, *rest in poly.exterior.coords]
    #         patch = MplPolygon(coords_2d, closed=True,
    #                         edgecolor='blue', facecolor='none', alpha=0.6) #lightblue
    #         ax.add_patch(patch)
    # else:
    #     ax.text(0.5, 0.5, "Unsupported Geometry Type", ha='center', va='center', transform=ax.transAxes)

    # ax.set_title("Rasterized Mask with Polygon Overlay")
    # ax.set_xlim(BBOX[0], BBOX[2])
    # ax.set_ylim(BBOX[1], BBOX[3])
    # ax.set_aspect('equal')

    # plt.show()
    # === Added plotting part ends here ===



    return mask




def run(kmz_path, data_directory, sb40_landfire, CBD_path, CH_path, CBH_path, ash_precent, sf_fl_factor, cp_fl_factor):

    processed_data_directory = os.path.join(data_directory, "processed_data")
    if not os.path.exists(processed_data_directory):
        os.mkdir(processed_data_directory)

    integrated_shp_file = ir_processing.integrate(kmz_path, os.path.join(processed_data_directory, "integrated_shp"))
    surface_fuel_load_map_path = fuel_load_map.calc_surface(sb40_landfire, processed_data_directory)
    canopy_fuel_load_map_path, canopy_height_map_path = fuel_load_map.calc_canopy(CBD_path, CH_path, CBH_path, processed_data_directory)

    surface_fl_ds = gdal.Open(surface_fuel_load_map_path)
    canopy_fl_ds = gdal.Open(canopy_fuel_load_map_path)
    canopy_h_ds = gdal.Open(canopy_height_map_path)


    geotransform_sf = surface_fl_ds.GetGeoTransform()
    geotransform_cp = canopy_fl_ds.GetGeoTransform()

    surface_fl = surface_fl_ds.GetRasterBand(1).ReadAsArray()
    canopy_fl = canopy_fl_ds.GetRasterBand(1).ReadAsArray()
    canopy_h = canopy_h_ds.GetRasterBand(1).ReadAsArray()


    RasterYSize_sf = surface_fl_ds.RasterYSize
    RasterXSize_sf = surface_fl_ds.RasterXSize

    RasterYSize_cp = canopy_fl_ds.RasterYSize
    RasterXSize_cp = canopy_fl_ds.RasterXSize

    xCellSize_sf = geotransform_sf[1]
    yCellSize_sf = -geotransform_sf[5]

    xCellSize_cp = geotransform_cp[1]
    yCellSize_cp = -geotransform_cp[5]

    BBOX_sf = [geotransform_sf[0], geotransform_sf[3]-RasterYSize_sf*yCellSize_sf, geotransform_sf[0]+RasterXSize_sf*xCellSize_sf, geotransform_sf[3]]    # [minx, miny, maxx, maxy]
    BBOX_cp = [geotransform_cp[0], geotransform_cp[3]-RasterYSize_cp*yCellSize_cp, geotransform_cp[0]+RasterXSize_cp*xCellSize_cp, geotransform_cp[3]]    # [minx, miny, maxx, maxy]

    data = gpd.read_file(integrated_shp_file)

    dataUTM10 = data.to_crs("EPSG:26910")


    time = []
    fuelLoad_sf_list = []
    fuelLoad_cp_list = []
    avgHeigth_cp_list = []
    area = []
    date = []

    corruptedLyr = False

    for lyr in range(len(dataUTM10)-1):

        if corruptedLyr:
            corruptedLyr = False
            continue

        print(dataUTM10.iloc[lyr]["name"])
        targetLyr = dataUTM10.iloc[lyr+1]["geometry"]
        previousLyr = dataUTM10.iloc[lyr]["geometry"]

        targetArea = targetLyr.area
        previousArea = previousLyr.area

        targetLyrDate = dataUTM10.iloc[lyr+1]["date"]
        targetLyrTime = dataUTM10.iloc[lyr+1]["time"]

        if targetArea<previousArea:
            corruptedLyr = True
            targetLyr = dataUTM10.iloc[lyr+2]["geometry"]
            targetLyrDate = dataUTM10.iloc[lyr+2]["date"]
            targetLyrTime = dataUTM10.iloc[lyr+2]["time"]

        targetArea = targetLyr.area
        previousArea = previousLyr.area

        area.append(targetArea)

        previousLyrDate = dataUTM10.iloc[lyr]["date"]
        previousLyrTime = dataUTM10.iloc[lyr]["time"]

        date.append(targetLyrDate + " " + targetLyrTime)


        targetLyrDateTime = datetime.strptime(f"{targetLyrDate} {targetLyrTime}", "%Y-%m-%d %H:%M")
        previousLyrDateTime = datetime.strptime(f"{previousLyrDate} {previousLyrTime}", "%Y-%m-%d %H:%M")

        poly = targetLyr.difference(previousLyr)
        time.append((targetLyrDateTime - previousLyrDateTime).total_seconds())

        
        if poly.area == 0:
            fuelLoad_sf_list.append(0)
            fuelLoad_cp_list.append(0)
            avgHeigth_cp_list.append(0)
        else:
            # mask_sf = extract(BBOX_sf, xCellSize_sf, yCellSize_sf, poly, 1)
            # print(100*np.sum(mask_sf)/(mask_sf.shape[0]*mask_sf.shape[1]))

            # import pylab as plt

            # # Plot the array
            # plt.imshow(mask_sf, cmap='viridis')  # You can use other colormaps like 'gray', 'jet', etc.
            # plt.colorbar()  # Optional: show color scale
            # plt.title("2D Array Plot")
            # plt.xlabel("X-axis")
            # plt.ylabel("Y-axis")
            # plt.show()
            # sys.exit()

            mask_sf = extract(BBOX_sf, xCellSize_sf, yCellSize_sf, poly, 67676)
            print("Extracting Surface Fuel Data ...")
            fuelLoad_sf_list.append(np.sum(surface_fl[mask_sf==67676])*30*30)

            mask_cp = extract(BBOX_cp, xCellSize_cp, yCellSize_cp, poly, 67676)
            print("Extracting Canopy Fuel Data ...")
            fuelLoad_cp_list.append(np.sum(canopy_fl[mask_cp==67676])*30*30)
            avgHeigth_cp_list.append(np.mean(canopy_h[mask_cp==67676]))




    df = pd.DataFrame()
    df["date"] = date
    df["sf_fl_t (kg)"] = fuelLoad_sf_list
    df["cp_fl_t (kg)"] = fuelLoad_cp_list
    df["cp_h_avg"] = avgHeigth_cp_list
    df["time (s)"] = time
    df["area (s)"] = area


    df.to_csv(os.path.join(data_directory, "processed_data", "fuel_time_fireArea_data.csv"))

    # filtering data
    df_filttered_data = df[df["sf_fl_t (kg)"] != 0]

    # calculating total fuel load
    total_fuel_load = sf_fl_factor*np.array(df_filttered_data["sf_fl_t (kg)"]) + cp_fl_factor*np.array(df_filttered_data["cp_fl_t (kg)"])
    # HARD CODE :: hard code. why? observation started in Aug. 26 and then we want to adjust the duration of the model with the observation data period. Since first step is for three days, we divided the values with 3
    total_fuel_load[0] = total_fuel_load[0]/3

    # calculating ash
    total_ash = total_fuel_load * ash_precent   #0.05

    # calculating sensible heat
    hfx_mj = total_fuel_load * 18.6
    time_arr = np.array(df_filttered_data["time (s)"])
    # HARD CODE :: hard code. why? observation started in Aug. 26 and then we want to adjust the duration of the model with the observation data period. Since first step is for three days, we divided the values with 3
    time_arr[0] = time_arr[0]/3
    hr_kw = (hfx_mj * 1000)/time_arr
    q_dot_kw = hr_kw*0.6

    df_filttered_data["total fuel"] = list(total_fuel_load)
    df_filttered_data["total ash (kg)"] = list(total_ash)
    df_filttered_data["HFX (MJ)"] = list(hfx_mj)
    df_filttered_data["modified_time (s)"] = list(time_arr)
    df_filttered_data["HR (KW)"] = list(hr_kw)
    df_filttered_data["q_dot (KW)"] = list(q_dot_kw)

    df_filttered_data.to_csv(os.path.join(data_directory, "processed_data", "ash_heat_data.csv"))

    return total_ash, q_dot_kw # more if you needed




