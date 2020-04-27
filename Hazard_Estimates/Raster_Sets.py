from Hazard_Estimates import raster_files as rf
import pandas as pd
import numpy as np


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
