
import pandas as pd
import numpy as np
import random as rand
import Hazard_Estimates.metrics as metrics
from copy import *

class X_Y:
    def __init__(self):
        """Independent and dependent variables"""

    def flood_hazard_dataset_setup(self, data_structure, aggregated, point):
        '''
        Get Flood Hazard Sample
        Add XY Values
        :param data_structure: Model data structure.
        :param aggregated: aggregated dataframe. Used for non-flooded areas
        :param point: Point based data frame. Used for Flooded areas.

        '''
        dataset = self.get_flood_hazard_sample(data_structure, aggregated, point)

        self.Add_XY_Values(data_structure, dataset)
        self.discrete = True

    def get_flood_hazard_sample(self, data_structure, df, claims_df):
        '''

        :param data_structure: Model structure
        :param df:  DataFrame
        :param claims_df:  Claims Dataframe
        :return: 1-1 Sample of of flooded non-flooded structures.
        '''
        unique_attoms = df["attom"].unique()
        sample = claims_df.loc[(claims_df["attom"].isin(unique_attoms)) & (claims_df.adj_damage > 0)]
        sample = sample.rename(columns={'x': "X", 'y': 'Y', })

        structure_sample = df.loc[df[data_structure.YColumn] == 0]
        columns = ['X', 'Y', data_structure.YColumn, 'year_of_loss', 'huc8']
        structure_sample = structure_sample.sample(n=len(sample), replace=True, random_state=42)
        structure_sample['year_of_loss'] = structure_sample.apply(
            lambda x: rand.randint(sample.year_of_loss.min(), sample.year_of_loss.max()), axis=1)

        return self.create_categorical_samples(pd.concat([sample[columns], structure_sample[columns]]), data_structure.YColumn, True)

    def flood_event_dataset_setup(self, data_structure, aggregated, hazardStructure, ):
        '''
        Similar to Flood_hazard_dataset_setup. Uses one Dataframe for flooded/ non-flooded.
        :param data_structure:  Model structure
        :param aggregated:      Aggregated DataFrame
        :param hazardStructure: Model Structure from flood hazard used to predict flood hazard on new sample
        :return:
        '''
        dataset = self.create_categorical_samples(aggregated, data_structure.YColumn, False)
        self.Add_XY_Values(data_structure, dataset)

        prob = hazardStructure.model.predict_proba(self.X_[hazardStructure.XColumns])
        self.X_['flood_prob'] = metrics.split_probabilities(prob)
        self.discrete = True


    def Add_XY_Values(self, data_structure, dataset, ):
        '''
        For each spatial index load rasters  and get columns.
        Make Sure column values are greater than 0.
        Make y binary.
        :param data_structure:  Model structure
        :param dataset: Sample of flooded non-flooded structures
        :return:  X Y data for model
        '''
        s_index_series = dataset[data_structure.Spatial_Index]
        if np.array_equal( s_index_series, s_index_series.astype(int)):
                self.X_ = pd.concat(
                [data_structure.iterate_rasters(dataset.loc[s_index_series == int(s_index)],
                                                int(s_index)) for s_index in
                 s_index_series.unique()])
        else:
            s_index: str
            self.X_ = pd.concat(
                [data_structure.iterate_rasters(dataset.loc[s_index_series == s_index],
                                                s_index) for s_index in
                 s_index_series.unique()])
        for column in data_structure.XColumns:
            try:
                self.X_ = self.X_.loc[self.X_[column] >= 0]
            except:
                continue
        self.Y_ = pd.DataFrame()
        self.Y_['inundated'] = np.where(self.X_[data_structure.YColumn] > 0, 1, 0)

    def create_categorical_samples(self, df, column, replacement):
        '''
        Categorical data pipeline
        :param df: dataframe
        :param column: YColumn
        :param replacement: WHether samples are creeated with replacement.
        :return: a 1-1 Sample of flooded and non-flooded structures.
        '''
        presence_dataset = df.loc[df[column] > 0]
        absence_dataset = df.loc[df[column] == 0].sample(n=len(presence_dataset), replace=replacement, random_state=42)

        return pd.concat([presence_dataset, absence_dataset])

    def exposure_dataset_setup(self, exposure_structure, subset=False, loc=[]):
        '''
        Subset variables if necessory, create dummies, combine everything.
        :param exposure_structure: Flood Exposure data structure
        :param subset: Boolean of whether to subest the model.
        :param loc: Boolean Array for subest
        :return: List of new columns.
        '''

        if subset == True:
            self.X_ = self.X_.loc[loc]

        dataset_dummies = pd.get_dummies(self.X_[exposure_structure.dummies])
        self.X_ = pd.concat([self.X_, dataset_dummies], axis=1, sort=False)

        self.discrete = False
        return dataset_dummies.columns

    def add_X_Y(self, x, y, discrete):
        '''
        If model is already configured this will add the variables where they need to be.
        :param x: Drivers
        :param y: Response variable
        :param discrete: Whether a categorical model should be used or a continuous regression.

        '''
        self.X_ = deepcopy(x)
        self.Y_ = deepcopy(y)
        self.discrete = discrete
