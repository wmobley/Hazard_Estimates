import os
import gdal

gdal.SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE', '7000')

directory = r"C:\Users\wmobley\Downloads\FloodHazard\Figures"


gdal.TranslateOptions( format ="COG",
                       )
gdal.Translate(os.path.join(directory, f"damagePlainCOG.tif"),
               os.path.join(directory, f"FloodHazard.tif")
               )
print("ran")