import time

from Hazard_Estimates.Model import *
from matplotlib import pyplot
import gc
import rasterio
hucNumber = "_Houston"
output = []

huc = "H:\HDM_Data\Spatial_Index/huc{}/".format(hucNumber)
files = [
    'demFill',
    # 'impervious',
    'TWI',
    'hand',
    'Distance2Coast',
    'distance2Stream',
    'AccumulatedFlow',
    'AverageKSAT',

]
dynamic_raster = [
            {'filename':'AverageRoughness',
                          'time':[ 2016]},
        {'filename': 'impervious',
         'time': [2016]}
    ]
files = [f.lower() for f in files]

raster_location = "H:\HDM_Data\Spatial_Index"
new_model = model_framework('Load Model', "huc", 'inundated', XColumns=files, file_location=raster_location)
new_model.Dynamic_rasters=dynamic_raster
new_model.load_model(r"D:\OneDrive_Tamu\OneDrive - Texas A&M University\NED\GLO\GLO_M3FR\Python\flood_hazard.rf")
new_model.model.n_jobs = -1

y = 2017
folder = r"D:\OneDrive_Tamu\OneDrive - Texas A&M University\NED\GLO\GLO_M3FR\Python\RFOutPut\rasterio"

for hucNumber in [
    # 12100407,
    #             12040104,
    #             12040204,
    #             12040101,
    #             12040102,
    #             12040202,
    #             12040203,
    #             12030203,
    #             12040103,
    #             12040201,
    #
    #             12090401,
    #             12070104,
                "huc12040104",]:
    raster_sets = new_model.locate_and_load(hucNumber)
    gc.collect()
    raster_sets.Convert_Dimensions()
    starti = time.time()
    src = raster_sets.generate_probability_raster(new_model,
                                            location=f"{folder}/FloodProbability{hucNumber}_2016",
                                            ignore_column='demfill',
                                            nodata=32767
                                            , file=f"H:/HDM_Data/Spatial_Index/huc{hucNumber}/demFill.tfw")
    print(f"Time per image: {time.time() - starti}")



    pyplot.imshow(src.asciiFile, vmin=0,  vmax=.2,  cmap='Blues')
    print(np.max(src.asciiFile))
    pyplot.show()
