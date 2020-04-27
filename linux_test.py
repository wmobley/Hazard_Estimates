import numpy as np
from Hazard_Estimates.Model import *
import rasterio
import os
import time
import sklearn

import gc
files = [
    'demFill',
    'impervious',
    'TWI',
    'hand',
    'distance2coast',
    'distance2stream',
    'AccumulatedFlow',
    'AverageKSAT',
    'AverageRoughness',
]

# files = [f.lower() for f in files]

raster_location = "huc"
new_model = model_framework('Load Model', "huc", 'inundated', XColumns=files, file_location=raster_location)
new_model.load_model("model.rf")


y = 2017


for hucNumber in [    12040104]:
    raster_sets = new_model.locate_and_load(hucNumber)

    raster_sets.Convert_Dimensions()
    starti = time.time()
    gc.collect()
    raster_sets.generate_probability_raster(model=new_model.model,
                                            location="probaility",
                                            ignore_column='demFill',
                                            nodata=32767
                                            , file=os.path.join(raster_location,"demFill.tfw"))
    print(f"Time per image: {time.time() - starti}")