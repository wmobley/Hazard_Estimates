import raster_files as rf
import pandas as pd
import psutil
import numpy as np
import metrics
from copy import *

class raster_sets:
    def __init__(self, files, storm=""):
        self.storm = storm
        self.rasters = [self.load_rasters(file) for file in files]
        self.prune_falses()

    def prune_falses(self):
        self.rasters = [self.prefix(r) for r in self.rasters if r != False]
        return self.rasters

    def load_rasters(self, file):
        raster = rf.ascii_raster()

        return raster.load(file)

    def prefix(self, x):
        '''
        Removes prefix from columns based on rain event from rasters.
        :param x: Row and column location
        :return:
        '''
        if x.fileName.startswith(self.storm):
            x.fileName = x.fileName[len(self.storm):]

        return x
    def Convert_Dimensions(self):
        '''
        Convert from 1d to 2d or back
        :return:
        '''
        for raster in self.rasters:
            if len(raster.asciiFile.shape)== 2:
                raster.make_dataset_1D()
            else: raster.return_dataset_2d(raster.nrows)



    def make_dataframe(self):
        dataset = pd.DataFrame()
        for r in self.rasters:
            if r.fileName not in [0, '0']:
                dataset[r.fileName] = r.asciiFile
        return dataset

    def generate_probability_raster(self, model, location, ignore_column, nodata, file):
        df = self.make_dataframe()
        print(df.columns)
        predictions =  rf.ascii_raster()
        predictions.asciiFile = np.where(df[ignore_column] != nodata, model.predict_proba(df)[:, 1], -9999)
        predictions.return_dataset_2d(self.rasters[0].nrows)
        print()
        self.rasters[0].save_image(predictions.asciiFile,location, file)


    #
    # def image_loop(test, landcover_values, crs_values):
    #     """Loop through 1-D image arrays. Requires 100 iterations."""
    #     prediction = []
    #     interval = int(len(raster_sets.rasters[0].asciiFile) / 1000)
    #     sub = 0
    #     for subset in range(interval, len(raster_sets.rasters[0].asciiFile), interval):
    #         p = image(test, sub, subset, landcover_values, crs_values)
    #         sub += interval
    #         prediction = np.append(prediction, p)
    #         prediction = prediction.astype(float)
    #     p = image(test, sub, len(raster_sets.rasters[0].asciiFile), landcover_values, crs_values)
    #     prediction = np.append(prediction, p)
    #     prediction = prediction.astype(float)
    #
    #     return prediction




    #
    #
    # '''set up dummies and raster_sets.rasters for image creation'''
    #
    #
    # def image(test, from_i, to_i, landcover_values, crs_values):
    #     d = pd.DataFrame()
    #     for r in test:
    #         if r.fileName not in [0, '0']:
    #             d[r.fileName] = r.asciiFile[from_i: to_i]
    #             if r.fileName == "landcover":
    #                 landcover = r.asciiFile[from_i: to_i]
    #             if r.fileName == "crs":
    #                 CRS = r.asciiFile[from_i: to_i]
    #     extra_columns = []
    #     if "landcover" in files:
    #         dummy_landuse = pd.get_dummies(landcover, prefix="landcover_")
    #         extra_columns.extend(["landcover__{}".format(value) for value in landcover_values if
    #                               "landcover__{}".format(value) not in dummy_landuse.columns.values])
    #     else:
    #         dummy_landuse = pd.DataFrame()
    #     for e in extra_columns:
    #         dummy_landuse[e] = 0
    #     if "crs" in files:
    #         dummy_crs = pd.get_dummies(CRS, prefix="crs_")
    #         extra_columns.extend([f"{value}" for value in crs_values if
    #                               f"{value}" not in dummy_crs.columns.values])
    #         print(dummy_crs.columns)
    #     else:
    #         dummy_crs = pd.DataFrame()
    #     for e in extra_columns:
    #         dummy_landuse[e] = 0
    #     d_clean = pd.concat([
    #         dummy_crs,
    #         dummy_landuse,
    #         d], axis=1)
    # d_clean = d_clean[X_train.columns.values]
    #
    #     d_clean = d_clean.fillna(0)
    #     p = np.where(d_clean.demfill != 32767, rf_fit.predict_proba(d_clean)[:, 1], -9999)
    #     # rf_fit.predict_proba(d_clean)[:, 1]
    #     #
    #
    #     return p