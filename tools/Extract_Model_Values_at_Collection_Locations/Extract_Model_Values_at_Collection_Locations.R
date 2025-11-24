# Developed by Facundo Scordo, Research Assistance Professor @ UNR, Fall 2025; Email: scordo@agro.uba.ar

###### Code to extract final model raster values at the in-situ sampling locations #####

# ---- Setup ----
# If needed, install packages once:
# install.packages(c("readxl", "sf", "terra", "dplyr"))

library(readxl)
library(sf)
library(terra)
library(dplyr)

# ---- Paths ----
tif_files <- c(
  "ERA5_Mass_Distribution_Maps\rChar_150_ash_5_Dc_dyn.tif",
  "ERA5_Mass_Distribution_Maps\rChar_150_ash_7_Dc_dyn.tif",
  "ERA5_Mass_Distribution_Maps\rChar_200_ash_5_Dc_dyn.tif",
  "ERA5_Mass_Distribution_Maps\rChar_200_ash_7_Dc_dyn.tif",
  "ERA5_Mass_Distribution_Maps\rChar_300_ash_5_Dc_dyn.tif",
  "ERA5_Mass_Distribution_Maps\rChar_300_ash_7_Dc_dyn.tif",
  "ERA5_Mass_Distribution_Maps\rChar_150_ash_5_Dc_0.68.tif",
  "ERA5_Mass_Distribution_Maps\rChar_150_ash_7_Dc_0.68.tif",
  "ERA5_Mass_Distribution_Maps\rChar_200_ash_5_Dc_0.68.tif",
  "ERA5_Mass_Distribution_Maps\rChar_200_ash_7_Dc_0.68.tif",
  "ERA5_Mass_Distribution_Maps\rChar_300_ash_5_Dc_0.68.tif",
  "ERA5_Mass_Distribution_Maps\rChar_300_ash_7_Dc_0.68.tif"
)

excel_path <- "CollectionPointsCoordinates.xlsx"
output_csv <- ""  # "Collection_vs_Model_Used_ERA5_Statistical_Comparison.csv" or "Collection_vs_Model_Used_WeatherStations_Statistical_Comparison.csv"

# ---- Read Excel (first sheet) ----
pts_raw <- read_excel(excel_path, sheet = 1)

# Auto-detect lat/lon columns (update this part)
lat_candidates <- c("lat","latitude","Lat","LAT","y","Y","Lat_Decimal","Latitude_Decimal")
lon_candidates <- c("lon","long","longitude","Long","LONG","x","X","Lon_Decimal","Longitude_Decimal")

pick_col <- function(df, candidates) {
  cn <- names(df)
  for (cand in candidates) {
    matches <- cn[tolower(cn) == tolower(cand)]
    if (length(matches) > 0) return(matches[1])
  }
  return(NA_character_)
}

lat_col <- pick_col(pts_raw, lat_candidates)
lon_col <- pick_col(pts_raw, lon_candidates)

if (is.na(lat_col) || is.na(lon_col)) {
  stop("Couldn't find latitude/longitude columns automatically. Please set 'lat_col' and 'lon_col' to the correct column names.")
}

message("Using latitude column: ", lat_col)
message("Using longitude column: ", lon_col)

# Build sf points (assume WGS84 coords)
pts_sf <- st_as_sf(
  pts_raw,
  coords = c(lon_col, lat_col),
  crs = 4326,
  remove = FALSE
)

# We'll keep the extracted data here
extracted_df <- st_drop_geometry(pts_sf)

# Loop over each TIFF file and extract raster values
for (tif_file in tif_files) {
  message("Processing raster: ", tif_file)
  r <- rast(tif_file)
  
  # Reproject points if raster CRS is different (and not lat-long)
  if (!st_is_longlat(st_crs(r)) && st_crs(pts_sf) != st_crs(r)) {
    pts_proj <- st_transform(pts_sf, st_crs(r))
  } else {
    pts_proj <- pts_sf
  }
  
  v <- vect(pts_proj)
  vals <- terra::extract(r, v, method = "bilinear")
  
  # Remove ID column, rename extracted columns with raster filename prefix
  vals_data <- vals[, -1, drop = FALSE]
  colnames(vals_data) <- paste0(
    tools::file_path_sans_ext(basename(tif_file)), "_", colnames(vals_data)
  )
  
  # Bind extracted values to dataframe
  extracted_df <- cbind(extracted_df, vals_data)
}

# Save output
write.csv(extracted_df, output_csv, row.names = FALSE)
message("Saved: ", normalizePath(output_csv))

# Preview output
print(head(extracted_df, 10))