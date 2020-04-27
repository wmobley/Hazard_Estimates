import time

from Hazard_Estimates.Model import *

hucNumber = "_Houston"
output = []

huc = "H:\HDM_Data\Spatial_Index/resample/huc{}/".format(hucNumber)
files = [
    'demFill',
    'impervious',
    'TWI',
    'hand',
    'Distance2Coast',
    'distance2Stream',
    'AccumulatedFlow',
    'AverageKSAT',
    'AverageRoughness',
]

files = [f.lower() for f in files]

raster_location = "H:\HDM_Data\Spatial_Index/resample/huc"
new_model = model_framework('Load Model', "huc", 'inundated', XColumns=files, file_location=raster_location)
new_model.load_model("H:\Hazard_Estimates\Flood_Hazard_Models\model.rf")


y = 2017
folder = r"D:\OneDrive_Tamu\OneDrive - Texas A&M University\NED\GLO\GLO_M3FR\Python\RFOutPut\rasterio"

for hucNumber in [
    12040104]:
    raster_sets = new_model.locate_and_load(hucNumber)

    raster_sets.Convert_Dimensions()
    starti = time.time()
    raster_sets.generate_probability_raster(model=new_model.model,
                                            location="probaility",
                                            ignore_column='demfill',
                                            nodata=32767
                                            , file=f"H:/HDM_Data/Spatial_Index/resample/huc{hucNumber}/demFill.tfw")
    print(f"Time per image: {time.time() - starti}")
