############################################################
#
# Developed by Majid Bavandpour
# Ph.D. Student @ UNR <majid.bavandpour@gmail.com>
# Summer 2025
#
############################################################
#
#
# The functions in this file calculate surface and canopy fuel loads maps using LANDFIRE data
#
#
############################################################

from osgeo import gdal, osr
import pandas as pd
import numpy as np
import sys, os
import libs.utils as utils

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')


## Dead Load + Live Load
lookUpTable = { 91: 0,
                93: 0,
                98: 0,
                99: 0,
                101: (0.0224 + 0.0673),
                102: (0.0224 + 0.2242),
                103: (0.1121 + 0.3363),
                104: (0.0560 + 0.4259),
                105: (0.0897 + 0.5604),
                106: (0.0224 + 0.7622),
                107: (0.2242 + 1.2105),
                108: (0.3363 + 1.6364),
                109: (0.4483 + 2.0175),
                121: (0.0448 + 0.1121),
                122: (0.2242 + 0.1345),
                123: (0.1233 + 0.3250),
                124: (0.5156 + 0.7622),
                141: (0.1121 + 0.0336),
                142: (1.0088 + 0),
                143: (0.7734 + 0),
                144: (0.4932 + 0),
                145: (1.2778 + 0),
                146: (0.9751 + 0),
                147: (2.4659 + 0),
                148: (1.4123 + 0),
                149: (1.5580 + 0.3475),
                161: (0.5828 + 0.0448),
                162: (0.8967 + 0),
                163: (0.3363 + 0.1457),
                164: (1.0088 + 0),
                165: (2.4659 + 0),
                181: (1.5244 + 0),
                182: (1.3226 + 0),
                183: (1.2329 + 0),
                184: (1.3899 + 0),
                185: (1.8046 + 0),
                186: (1.0760 + 0),
                187: (2.1969 + 0),
                188: (1.8606 + 0),
                189: (3.1608 + 0),
                201: (3.4746 + 0),
                202: (2.8582 + 0),
                203: (2.5219 + 0),
                204: (3.1384 + 0) }



def calc_surface(sb40_landfire, fuel_load_map_folder):

    fuel_load_map_path = os.path.join(fuel_load_map_folder, "surface_fuel_load_map.tif")

    if os.path.exists(fuel_load_map_path):
        return fuel_load_map_path

    ds_fuel_map = gdal.Open(sb40_landfire)
    geotransform = ds_fuel_map.GetGeoTransform()
    fuelMapCode = ds_fuel_map.GetRasterBand(1).ReadAsArray()


    fuelMap = np.zeros(fuelMapCode.shape)
    fuelMap[fuelMapCode == -9999] = np.nan

    keys = list(lookUpTable.keys())
    for key in keys:
        fuelMap[fuelMapCode == key] = lookUpTable[key]

    
    utils.write_geotiff(fuelMap, geotransform, fuelMap.shape[1], fuelMap.shape[0], fuel_load_map_path, crs=None, wkt=ds_fuel_map.GetProjection())

    return fuel_load_map_path



def calc_canopy(CBD_path, CH_path, CBH_path, fuel_load_map_folder):

    fuel_load_map_path = os.path.join(fuel_load_map_folder, "canopy_fuel_load_map.tif")
    canopy_height_map = os.path.join(fuel_load_map_folder, "canopy_height_map.tif")

    if os.path.exists(fuel_load_map_path) and os.path.exists(canopy_height_map):
        return fuel_load_map_path, canopy_height_map

    CBD_ds = gdal.Open(CBD_path)
    CH_ds = gdal.Open(CH_path)
    CBH_ds = gdal.Open(CBH_path)

    geotransform = CBD_ds.GetGeoTransform()

    CBD = CBD_ds.GetRasterBand(1).ReadAsArray()/100
    CH = CH_ds.GetRasterBand(1).ReadAsArray()/10
    CBH = CBH_ds.GetRasterBand(1).ReadAsArray()/10

    CBD[CBD == 32767/100] = np.nan
    CH[CH == 32767/10] = np.nan
    CBH[CBH == 32767/10] = np.nan

    CFL = CBD*(CH-CBH)




    utils.write_geotiff(CFL, geotransform, CFL.shape[1], CFL.shape[0], fuel_load_map_path, crs=None, wkt=CBD_ds.GetProjection())
    utils.write_geotiff(CBH, geotransform, CBH.shape[1], CBH.shape[0], canopy_height_map, crs=None, wkt=CBD_ds.GetProjection())

    return fuel_load_map_path, canopy_height_map


