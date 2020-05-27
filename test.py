from copy import *

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from Hazard_Estimates import Model

def create_y(df, YColumn, category):
    return np.sqrt(df[YColumn] / df['no_of_structures'])

structures = pd.read_csv(r'D:\OneDrive_Tamu\OneDrive - Texas A&M University\NED\GLO\GLO_M3FR\Python\structures_nad.csv')
structures = structures.loc[structures.huc8==12040102]
structures = structures.rename(columns={"x": "X", "y": "Y", 'sum_dmg':"adj_damage"})
structures['occupancy']=structures['occupancy'].apply(lambda x: x[:4])

hucs = structures.loc[structures.adj_damage>0].huc8.unique()
structures= structures.loc[structures.huc8.isin(hucs)]
structures[[ 'structure', 'content']] = np.abs(structures[[ 'structure', 'content']])

# structures.loc[structures.greenspoin==1].to_csv("Greenspoint_structures.csv")
# structures.loc[structures.LakeJackso==1].to_csv("LakeJackson_structures.csv")
structures = structures.loc[(structures.greenspoin==0) &(structures.LakeJackso!=1)]
structures.drop(['greenspoin', 'LakeJackso'],axis=1, inplace=True)
structures.dropna(inplace = True)
structures_train, structures_test = train_test_split(structures, test_size=.3)
print(f" Percent of train for the 17 storm: {len(structures_train.loc[structures_train.sum_17>0])/ len(structures.loc[structures.sum_17>0]) *100} \nShould be approximately 70%")
claims = pd.read_csv(r"D:\OneDrive_Tamu\OneDrive - Texas A&M University\NED\GLO\GLO_M3FR\Python\claims_huc.csv")
claims= claims.loc[claims.huc8.isin(hucs)]


claims.x = abs(claims.x)
claims.y = abs(claims.y)
file_location = r"H:\HDM_Data\Spatial_Index\resample\huc"
k_fold = KFold(2, shuffle=True)
files_hazard = [
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
file_location = r"H:\HDM_Data\Spatial_Index\resample\huc"
flood_hazard = Model.model_framework('rf', "huc8", 'adj_damage', files_hazard, file_location)

files_event = [
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
flood_event = Model.model_framework('rf_event', "huc8", 'sum_17', files_event, file_location, "harvey")

dummy_vars = [
    # 'occupancy',
              'foundation', 'exterior', 'storey']
continuous = ['ffe', 'area',
              #              'AccumulatedFlow',
              'AverageKSAT',
              #              'AverageRoughness',
              'Distance2Coast', 'TWI',
              'demFill', 'impervious', 'distance2Stream', 'hand', "flood_prob",
              'event_prob',
              'structure', 'content',
              ]
continuous.extend([f"{hr}hr" for hr in [1, 2, 4, 8, 24, ]])

flood_exposure = Model.model_framework('rf_exposure', "huc8", 'sum_17', continuous, file_location, "harvey",
                                 split_model="occupancy", create_Y = create_y)
for k, (cv_train, cv_test) in enumerate(k_fold.split(structures_train)):

    flood_hazard.train.flood_hazard_dataset_setup(flood_hazard, structures_train.iloc[cv_train], claims)
    flood_hazard.test.flood_hazard_dataset_setup(flood_hazard, structures_train.iloc[cv_test], claims)
    flood_hazard.set_up_categorical()
    #### Flood Event
    flood_event.train.flood_event_dataset_setup(flood_event, structures_train.iloc[cv_train], flood_hazard)
    flood_event.test.flood_event_dataset_setup(flood_event, structures_train.iloc[cv_test], flood_hazard)
    flood_event.XColumns.extend([f"{hr}hr" for hr in [1, 2, 4, 8, 24, ]])
    flood_event.XColumns.append("flood_prob")
    flood_event.set_up_categorical()
    flood_event.train.X_['event_prob'] = flood_event.metrics.split_probabilities(
        flood_event.model.predict_proba(flood_event.train.X_[flood_event.XColumns]))
    flood_event.test.X_['event_prob'] = flood_event.metrics.split_probabilities(
        flood_event.model.predict_proba(flood_event.test.X_[flood_event.XColumns]))

    #### Flood Exposure
    flood_exposure.XColumns = continuous
    flood_exposure.dummies = dummy_vars
    flood_exposure.train = deepcopy(flood_event.train)
    flood_exposure.test = deepcopy(flood_event.test)

    loc = ((flood_exposure.train.X_['event_prob'] < .3) & (flood_exposure.train.X_[flood_exposure.YColumn] == 0)) | (
                flood_exposure.train.X_[flood_exposure.YColumn] > 0)

    flood_exposure.XColumns.extend(flood_exposure.train.exposure_dataset_setup(flood_exposure, subset=True, loc=loc))
    loc_y = (flood_exposure.test.X_[flood_exposure.YColumn] > 0)

    #     flood_exposure.XColumns.extend(flood_exposure.train.exposure_dataset_setup(flood_exposure))
    flood_exposure.test.exposure_dataset_setup(flood_exposure, subset=True, loc=loc_y)
    if k < 1:
        flood_exposure.set_up_continious()
    else:
        for category in flood_exposure.train.X_[flood_exposure.split_model].unique():
            flood_exposure.equalize_train_test_columns()

            y = flood_exposure.create_Y(flood_exposure.train.X_.loc[flood_exposure.train.X_[flood_exposure.split_model]==category], flood_exposure.YColumn, category)
            x = flood_exposure.train.X_[flood_exposure.XColumns].loc[flood_exposure.train.X_[flood_exposure.split_model]==category]
            flood_exposure.model[category].fit(x, y)
    flood_exposure.metrics.add_accuracies(flood_exposure, subset=False, test=True)

    print(flood_exposure.metrics.__dict__)