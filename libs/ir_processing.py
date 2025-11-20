############################################################
#
# Developed by Majid Bavandpour
# Ph.D. Student @ UNR <majid.bavandpour@gmail.com>
# Summer 2025
#
############################################################
#
#
# The functions in this file read the IR fire polygons (from .kmz files), preprocess, calculate fire area advances, and store in a .shp file
#
#
############################################################




import libs.ogr2ogr as ogr2ogr
import os, sys
from osgeo import ogr, osr
import warnings
warnings.filterwarnings("ignore")
import shutil

# sys.setrecursionlimit(10**6)

def createDS(ds_name, ds_format, geom_type, srs, overwrite=False):
    drv = ogr.GetDriverByName(ds_format)
    if os.path.exists(ds_name) and overwrite is True:
        files = os.listdir(ds_name)
        for file in files:
            os.remove(os.path.join(ds_name, file))
    ds = drv.CreateDataSource(ds_name)
    lyr_name = os.path.splitext(os.path.basename(ds_name))[0]
    lyr = ds.CreateLayer(lyr_name, srs, geom_type)

    nameField = ogr.FieldDefn("name", ogr.OFTString)
    lyr.CreateField(nameField)
    # dateTimeField = ogr.FieldDefn("datetime", ogr.OFTDateTime)
    # lyr.CreateField(dateTimeField)
    dateField = ogr.FieldDefn("date", ogr.OFTString)
    lyr.CreateField(dateField)
    timeField = ogr.FieldDefn("time", ogr.OFTString)
    lyr.CreateField(timeField)
    areaField = ogr.FieldDefn("area", ogr.OFTString)
    lyr.CreateField(areaField)

    return ds, lyr


def dedup(geometries):
    """Return a geometry that is the union of all geometries."""
    if not geometries:  return None
    multi  = ogr.Geometry(ogr.wkbMultiPolygon)
    for g in geometries:
        g.geometry().CloseRings()
        multi.AddGeometry(g.geometry())
    return multi.UnionCascaded()


def cleanDir(path):
	print ("####### Cleaning Directory #######")
	Files = os.listdir(path)
	for File in Files:
		if File.endswith(".shp"):
			if os.stat(os.path.join(path, File)).st_size <= 100:

				os.remove(os.path.join(path, File))
				os.remove(os.path.join(path, File[:len(File) - 4] + ".dbf"))
				os.remove(os.path.join(path, File[:len(File) - 4] + ".prj"))
				os.remove(os.path.join(path, File[:len(File) - 4] + ".shx"))

				print("File: " + File + " was deleted")
	
	print("####### Done #######")


def main(FileKMZ, FolderSHP):

  SHP_Folder_Name = os.path.basename(os.path.normpath(FileKMZ))
  SHP_Folder_Name = SHP_Folder_Name.strip('.kmz')
  SHP_Folder_Name = SHP_Folder_Name.strip('.kml')
  
  Out_Folder = os.path.join(FolderSHP, SHP_Folder_Name)
  
  print(Out_Folder)
  ogr2ogr.main(["", "-skipfailures", "-f", "ESRI Shapefile", Out_Folder, FileKMZ]) 


def run_kmz2shp(FolderKMZ, FolderSHP):

	files = os.listdir(FolderKMZ)

	for file in files:
		if file.endswith('.kmz') or file.endswith('.kml'):
			FileKMZ = os.path.join(FolderKMZ, file)
			try:
				main(FileKMZ, FolderSHP)
			except:
				pass


def union(layer):

	if layer.GetFeatureCount() > 1:
		union = dedup(layer)

		return union	
		
	else:

		feat_ = layer.GetNextFeature()
		geom = feat_.GetGeometryRef()
		print(type(geom))

		return geom


def integrate(kmz_path, integrated_shp_path):

	output_path = os.path.join(integrated_shp_path, os.path.basename(os.path.normpath(integrated_shp_path))+".shp")

	if os.path.exists(output_path):
		return output_path

	# creating shp file path if not exist
	if not os.path.exists(integrated_shp_path):
		os.mkdir(integrated_shp_path)

	# creating a temp folder for shps converted fom kmz
	temp_folder = os.path.join(kmz_path, "temp")
	if os.path.exists(temp_folder):
		shutil.rmtree(temp_folder)
		os.mkdir(temp_folder)
	else:
		os.mkdir(temp_folder)


	# First Step: Convert KMZ to SHPs ######
	run_kmz2shp(kmz_path, temp_folder)


	# Second Step: Integrating Heat Perimeter #######
	folders = os.listdir(temp_folder)


	for i, folder in enumerate(sorted(folders)):
		if folder != '.DS_Store':
			name = 'HeatPerimeter_TS' + "{:03d}".format(i)
			date = folder[0:4] + '-' + folder[4:6] + '-' + folder[6:8]
			time = folder[9:11] + ':' + folder[11:13]

			files = os.listdir(os.path.join(temp_folder, folder))
			shp_files = []
			shp_file = ""
			for file in files:
				if file.endswith('.shp'):
					if ('HeatPerimete' in file) or ('Heat_Perimieter' in file) or ('Heat Perimeter' in file) or ('HeatPerimeter' in file) or ('Heat_Perimeter' in file) or ('IR_Polygon' in file) or ('Caldor_IR' in file) or ("Heat" not in file) and ('Point' not in file):
						shp_files.append(os.path.join(temp_folder, folder, file))
						# print(file)
						
			if len(shp_files) == 0:

				print(name, date, time, ": Cannot find the file")

			else:
				for ii, shp_file in enumerate(shp_files):

					DataSet = ogr.Open(shp_file)
					lyr = DataSet.GetLayer()


					if i == 0 and ii == 0:
						out_ds, out_lyr = createDS(integrated_shp_path, DataSet.GetDriver().GetName(), lyr.GetGeomType(), lyr.GetSpatialRef(), True)
						defn = out_lyr.GetLayerDefn()


					# geom = union(lyr)
					if lyr.GetFeatureCount() ==0:
						continue

					if lyr.GetFeatureCount() > 1:
						union = dedup(lyr)
						
					else:

						feat_ = lyr.GetNextFeature()
						union = feat_.GetGeometryRef()


					out_feat = ogr.Feature(defn)



					out_feat.SetGeometry(union)

					out_feat.SetField("name", name)
					# out_feat.SetField("datetime", "2021/09/06 18:12:00")
					out_feat.SetField("date", date)
					out_feat.SetField("time", time)
					# out_feat.SetField("area", area)
					out_lyr.CreateFeature(out_feat)

					DataSet.Destroy()
					print(name, date, time, ": Done !")



	out_ds.Destroy()

	shutil.rmtree(temp_folder)

	return output_path
	

