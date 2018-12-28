# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 09:49:05 2018

@author: mmann
"""

import numpy as np
import glob
import os.path
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
import gdal
from re import sub
from pathlib import Path
import os.path
from tsraster.prep import image_to_array,image_to_series, image_names, if_series_to_df



import matplotlib.pyplot as plt
%matplotlib inline
from rasterio.plot import show
import tsraster
import tsraster.prep as tr

from tsraster.calculate import calculateFeatures
from tsraster.calculate import features_to_array

#%% Set example data paths

ex_crs = r'F:\3month'
ex_ts = r'F:\3month_ts'
ex_single = r'F:/3month_ts/aet/aet-201201.tif'

tg_crs = r'F:\3month_fire'
tg_ts = r'F:\3month_ts_fire'
tg_single = r'F:\3month_ts_fire\fire_1985.tif'
mask = r'F:\Boundary\StatePoly_buf.tif'

#%% image_names PASS

print(tr.image_names(ex_crs))
print(tr.image_names(ex_ts))
print(tr.image_names(ex_single))


#%% read_images PASS

print(tr.read_images(ex_crs))
print(tr.read_images(ex_ts))
tr.read_images(ex_single)

#%% image_to_array PASS

print(tr.image_to_array(ex_crs).shape)
print(tr.image_to_array(ex_ts).shape)
print(tr.image_to_array(ex_single).shape)


#%% image_to_series PASS 

print(tr.image_to_series(ex_crs).head())
print( tr.image_to_series(ex_ts).head())
print( tr.image_to_series(ex_single).head())

#%% image_to_series for target data (PASS)

print(tr.image_to_series(tg_crs).head())
print( tr.image_to_series(tg_ts).head())
print( tr.image_to_series(tg_single).head())

 

#%% mask unmask PASS I think  mk_ex_crs might have too many observations, but can't figure out how

df_ex_crs = tr.image_to_series(ex_crs)
df_ex_ts = tr.image_to_series(ex_ts)
df_ex_single = tr.image_to_series(ex_single)


#original size 
print(df_ex_single.shape)

# masked 
mk_ex_crs = tr.mask_df(raster_mask= mask, original_df = df_ex_crs,missing_value = -9999)
mk_ex_ts = tr.mask_df(raster_mask= mask, original_df = df_ex_ts,missing_value = -9999)
mk_ex_single = tr.mask_df(raster_mask= mask, original_df = df_ex_single,missing_value = -9999)
 
print(mk_ex_crs.shape[0]/3 ) # three periods PROBLEM TOO MANY OBSERVATIONS
print(mk_ex_ts.shape[0]/6 ) # three periods for two variables 
print(mk_ex_single.shape)


#%% unmask PASS

print(df_ex_crs.shape)
print(tr.unmask_df(original_df= df_ex_crs, mask_df_output= mk_ex_crs).shape)
print(df_ex_ts.shape)
print(tr.unmask_df(original_df= df_ex_ts, mask_df_output= mk_ex_ts).shape)
print(df_ex_single.shape)
print(tr.unmask_df(original_df= df_ex_single, mask_df_output= mk_ex_single).shape)


#%% check_mask PASS

tr.check_mask(raster_mask = mask, raster_input_ex = ex_single)


#%% calculateFeatures

import os.path
import matplotlib.pyplot as plt
%matplotlib inline
import tsraster
import tsraster.prep as tr
from math import ceil
import os.path

import matplotlib.pyplot as plt
%matplotlib inline
from rasterio.plot import show
import tsraster
import tsraster.prep as tr

from tsraster.calculate import calculateFeatures
from tsraster.calculate import features_to_array


path = ex_crs
 


fc_parameters = {
    "mean": None,
    "maximum": None,
    "minimum":None
 }

ts_features = calculateFeatures(path,
                                parameters=fc_parameters,
                                reset_df=True,
                                tiff_output=True)
#%%

import numpy as np
import pandas as pd
import os
import gdal
import glob
from pathlib import Path
from tsfresh import extract_features
from tsfresh.utilities.distribution import MultiprocessingDistributor, LocalDaskDistributor
from tsfresh.feature_selection.relevance import calculate_relevance_table as crt
from tsraster.prep import image_to_series, image_to_array, read_images

parameters=fc_parameters
reset_df=False
tiff_output=True

#def calculateFeatures(path, parameters, reset_df, tiff_output=True):
'''
Calculates features or the statistical characteristics of time-series raster data.
It can also save features as a csv file (dataframe) and/or tiff file.

:param path: directory path to the raster files
:param parameters: a dictionary of features to be extracted
:param reset_df: boolean option for existing raster inputs as dataframe
:param tiff_output: boolean option for exporting tiff file
:return: extracted features as a dataframe and tiff file
'''
  
if reset_df == False:
    #if reset_df =F read in csv file holding saved version of my_df
	    my_df = pd.read_csv(os.path.join(path,'my_df.csv'))
else:
    #if reset_df =T calculate ts_series and save csv
    my_df = image_to_series(path)
    print('df: '+os.path.join(path,'my_df.csv'))
    my_df.to_csv(os.path.join(path,'my_df.csv'), chunksize=10000, index=False)

Distributor = MultiprocessingDistributor(n_workers=6,
                                         disable_progressbar=False,
                                         progressbar_title="Feature Extraction")


extracted_features = extract_features(my_df, 
                                      default_fc_parameters=parameters,
                                      column_sort="time",
                                      column_value="value",
                                      column_id=my_df.index(level="index"),
                                      distributor=Distributor
                                      )

# deal with output location 
out_path = Path(path).parent.joinpath(Path(path).stem+"_features")
out_path.mkdir(parents=True, exist_ok=True)

# get file prefix
if os.path.isdir(path):
    prefix = os.path.basename(glob.glob("{}/**/*.tif".format(path), recursive=True)[0])[0:4]
    
# write out features to csv file
print("features:"+os.path.join(out_path,'extracted_features.csv'))
extracted_features.columns = [prefix + str(col) for col in extracted_features.columns]
extracted_features.to_csv(os.path.join(out_path,'extracted_features.csv'), chunksize=10000)

# write data frame
kr = pd.DataFrame(list(extracted_features.columns))
kr.index += 1
kr.index.names = ['band']
kr.columns = ['feature_name']
kr.to_csv(os.path.join(out_path,"features_names.csv"))

# write out features to tiff file
if tiff_output == False:

    '''tiff_output is true and by default exports tiff '''

    return extracted_features
else:
    # get image dimension from raw data
    rows, cols, num = image_to_array(path).shape
    # get the total number of features extracted
    matrix_features = extracted_features.values
    num_of_layers = matrix_features.shape[1]
    
    #reshape the dimension of features extracted
    f2Array = matrix_features.reshape(rows, cols, num_of_layers)
    output_file = 'extracted_features.tiff'  
    
    #Get Meta Data from raw data
    raw_data = read_images(path)
    GeoTransform = raw_data[0].GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')
    
    noData = -9999
    
    Projection = raw_data[0].GetProjectionRef()
    DataType = gdal.GDT_Float32
    
    #export tiff
    CreateTiff(output_file, f2Array, driver, noData, GeoTransform, Projection, DataType, path=out_path)
    return extracted_features


#%%
path = ex_ts

ts_features = calculateFeatures(path,
                                parameters=fc_parameters,
                                reset_df=True,
                                tiff_output=True)

#%%


