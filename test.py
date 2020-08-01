import gc
import time

import sys

from sklearn.model_selection import train_test_split

from Hazard_Estimates.Model import *
import multiprocessing

hucNumber = 'huc12110205'
n_jobs = -1






def create_y(df, YColumn, category):
    return (df[YColumn] / df['no_of_structures']).where(df[category] == 'RES3', df[YColumn])


def rescale_y(y, x, category):
    return ((y) * x['no_of_structures']).where(x[category] == 'RES3', y)


files_hazard = [
'dem',
 'TWI',
 'hand',
 'Distance2Coast',
 'Distance2Streams',
 'Distance2Lakes',
 'FlowAccumulation',
 'AverageKSAT',
 'Precip25_12ha',
 'Precip25_60ma',
 'Precip100_12ha',
 'Precip100_60ma',
 'Precip500_12ha',
 'Precip500_60ma'

    ]
dynamic_raster = [
        #     {'filename':'AverageRoughness',
        #                   'time':2016},
        {'filename': 'impervious',
         'time': 2016},
    ]
file_location = r"H:\HDM_Data\Spatial_Index"
flood_hazard = model_framework( 'rf', "huc8",'adj_damage',files_hazard, file_location  )
flood_hazard.load_model( r'D:\OneDrive_Tamu\OneDrive - Texas A&M University\NED\GLO\GLO_M3FR\Python\LargeModel.rf')
flood_hazard.max = 2017
flood_hazard.min = 1976
flood_hazard.Dynamic_rasters = dynamic_raster

for y_column, storm, model in [( 'sum_17', 'Hanna',r'D:\OneDrive_Tamu\OneDrive - Texas A&M University\NED\GLO\GLO_M3FR\Python\LargeModel.rf')] :


    print(hucNumber)

    # if os.path.exists(f"harvey/flood_hazard{hucNumber}.tif"): exit()
    raster_sets = flood_hazard.locate_and_load(hucNumber)
    for r in raster_sets.rasters:
        if r.fileName =='hand':
            r.asciiFile = np.where(r.asciiFile==r.src._nodatavals, -9999, r.asciiFile)
        r.asciiFile = np.nan_to_num(r.asciiFile, nan=0)
        print(r.fileName)
    gc.collect()
    raster_sets.Convert_Dimensions()

    starti = time.time()
    flood_hazard.model.n_jobs = n_jobs
    raster_sets.rasters.append(raster_sets.generate_probability_raster(flood_hazard,
                                                                       location=f"{storm}/FP_{hucNumber[3:]}",
                                                                       ignore_column='hand',
                                                                       nodata=-9999
                                                                       ,
                                                                       file=f"{hucNumber}/dem.tfw",
                                                                       annualize = True
                                                                       ),
                               )

    gc.collect()
    # if os.path.exists(f"harvey/{hucNumber}.tif"): exit()

    # raster_sets.rasters[-1].fileName = "flood_prob"
    # raster_sets.rasters[-1].make_dataset_1D()
    # raster_sets.generate_probability_raster(flood_event,
    #                                         location=f"{storm}/{storm}_{hucNumber[3:]}",
    #                                         ignore_column='demFill',
    #                                         nodata=32767
    #                                         , file=f"{hucNumber}/demFill.tfw")
    gc.collect()
    print(f"{hucNumber} minutes: {(time.time() - starti) / 60}")
