
############################################################
#
# Developed by Majid Bavandpour
# Ph.D. Student @ UNR <majid.bavandpour@gmail.com>
# Summer 2025
#
############################################################
#
#
# Function to write out a raster file
#
#
############################################################


from osgeo import gdal, osr
import numpy as np




def write_geotiff(array, geotransform, Xsize, Ysize, tif_file_path, crs=None, wkt=None):

    # reload(gdal)
    # reload(osr)

    driver = gdal.GetDriverByName("GTiff")
    
    # array[array<=0] = np.nan

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


