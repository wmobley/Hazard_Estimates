
# Hazards Estimates library

Framework for using ML classification models on point locations across rasters. 

## Dependencies
- Fiona
- rasterio
- pandas
- scikit-learn (>=0.22.2post1)
- joblib
- numpy (>=1.22.0)

## Installation

The easiest way to install this repository is to use **conda**.
Download the environment.yml file add it to your project and then use the following lines:

```
conda env create -f environment.yml
conda activate gdal

python -m ipykernel install --user --name 
gdal  --display-name "GDAL"
```

While this approach isn't completely necessary, it does provide more functionality. The environment is GDAL friendly which is always a pain. 

If you don't want all the extra software overhead and just want a simple install of hazard estimates: 

```
pip install -U git+https://github.com/wmobley/Hazard_Estimates.git
```


## Getting Started

I initially wrote the hazards estimates to provide a simplified entry point for species distribution modelling (SDM) if the model accounts for dynamic explanatory features. I'll start off by showing how to create a basic SDM and then how we can account for dynamic landscapes. 

### Import Libraries
For the basic run we need to import two libraries, Hazard_Estimates and Pandas, the hazard Estimates does the rest of the work. 

```
from Hazard_Estimates.Hazard_Estimates.Model import *
import pandas as pd

```

### Sample Points
From here we need to load presence and background locations. I've kept the background points fairly vague. This allows you to select how you want to sample your background points. 


```
presence    = pd.read_csv("presence.csv")
background = pd.read_csv("absence.csv")
```

You'll  need a few columns and to configure a few columns in the model below. 
- **X** (Float): X location of the point
- **Y** (Float): Y Location of the point
- **year_of_loss** (Int): This column identifies when the Presence samples were taken. It is used for the dynamic landscapes. If you aren't using dynamic landscapes you can have any integer here. 
- **Presence** (Int): The model doesn't require you to have a specific column name here. What you need is a column with 1's for the presence and the same named column with 0's for the background or pseudo-absences. 

- **Spatial Index** "String": This column is only necessary if you are using large images. If the images are too large they can't all be held in memory. Breaking them down into smaller locations with a spatial index can improve performance. For small models use `None` otherwise add the boundary for each spatial indexed raster here. *See spatial index section on how to organize this.*  



### Model Configurations
We now need to set up the model configuration. This includes: 
- The raster names used as columns
- The rasters that are considered dynamic, and what years they cover *If there are no dynamic rasters you can remove this variable*
- The raster extension: ".tif", ".vrt"
- And the locations of the raster files. 

```
raster_file_names = [
            'dem',
            'hand',
            'Distance2Coast',
            'Distance2Streams',
            'FlowAccumulation',
            'AverageKSAT',
       
]
dynamic_rasters = [{'filename':'AverageRoughness',
                  'time':[2016]},
                 {'filename':'impervious',
                  'time':[2016]},
                ]
file_location = r"location of the vrt files"
```




### Initialize Model Framework
Now we will initialize the framework, and set up the configurations previously mentioned. 
```
flood_hazard = model_framework( 
								model_name='rf', 
								spatialIndex=None, 
								ycolumn='Presence',
								XColumns=raster_file_names, 
								file_location=file_location  )
flood_hazard.Dynamic_rasters = dynamic_rasters
```
Make sure that all of your rasters have the same extension. The raster_file_names will become the columns in the pandas tables. 
```
flood_hazard.set_raster_extension(".vrt")
```

###
Once all the configurations are set up we can set up the flood hazard model training data:
```

flood_hazard.train.flood_hazard_dataset_setup(
									data_structure = flood_hazard,
									presence=presence, 
									background=background )
```

Then you can fit a model as follows: 

```
flood_hazard.set_up_categorical()
```

The set_up_categorical function has one optional variable you can assign an sklearn model such as a random forest, boosted trees etc  to the variable for a customized model. Currently it defaults to a calibrated random forest model for flooding. 

```
from sklearn.ensemble import  GradientBoostingClassifier
gradient_boosting = GradientBoostingClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=10
)
flood_hazard.set_up_categorical(model = gradient_boosting)
```

### Predicting Images

You can then predict full images using the following code: 

```
raster_sets = flood_hazard.locate_and_load(spatial_index="")
```
The locate_and_load function loads all of the rasters you've identified for the framework into memory. If you are using a spatial index column in your training data. Add it as a variable otherwise leave it as an empy string. 


```
raster_sets.generate_probability_raster(
											model_structure=flood_hazard,
											location="location to save file",
											ignore_column='dem',
											nodata=32767,
											file=f"H:/HDM_Data/Spatial_Index/huc{hucNumber}/demFill.tfw")
```

- model_structure: the framework intialized previously
- location: location where you want the file saved
- Ignore_Column: which image you want to base your ignore data on. 
- nodata: the number you wnat to use for nodata
