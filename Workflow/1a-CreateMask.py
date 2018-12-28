# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:33:32 2018

@author: mmann
"""

#%%

import tsraster.prep as tr
import tsraster.calculate as ca
import pandas as pd

#%%

# rasterize the state polygon to match inputs 
tr.poly_rasterizer(poly = 'F:/Boundary/StatePoly.shp', 
                   raster_ex = r'F:/5year/aet/aet-201201.tif',
                   raster_path_prefix = r'F:/Boundary/StatePoly_buf',
                   buffer_poly_cells=10)


#%%  CHECK MASK
raster_mask = u"F:/Boundary/StatePoly_buf.tif"
raster_input_ex = u"F:/5year/aet/aet-201201.tif"

tr.check_mask(raster_mask,raster_input_ex)



#%% Prep a dataframe

#path = r"F://3month/"
#mask =  r"F:/Boundary/StatePoly_buf.tif"
#parameters = {
#    "mean": None,
#    "maximum": None}
#
#extracted_features = ca.calculateFeatures(path, 
#                                          parameters, 
#                                          reset_df=False, 
#                                          tiff_output=False)
#extracted_features.head()

#%%Mask a dataframe

raster_mask = u"F:/Boundary/StatePoly_buf.tif"
original_df = r"F:\3month_features\extracted_features.csv"
df_mask  = tr.mask_df(raster_mask, original_df)


#%% Mask a series 
raster_mask = u"F:/Boundary/StatePoly_buf.tif"
original_series  = tr.image_to_series_simple(raster_mask)
series_mask =  tr.mask_df(raster_mask, original_series)


#%% mask a long format dataframe
path = r"F://3month/"
my_df = tr.image_to_series(path)
raster_mask =  r"F:/Boundary/StatePoly_buf.tif"
long_mask =  tr.mask_df(raster_mask, my_df)


#%% Test unmask for series
# Update the values in the masked dataset to something else, here 10
series_mask.iloc[:] = 10

# unmask and update values of original df
print(original_series.iloc[:].value_counts().head(4))
updated_s = tr.unmask_df(original_series, series_mask)
print(updated_s.iloc[:,0].value_counts().head(4))
print(original_series.shape)
print(series_mask.shape)
print(updated_s.shape)
updated_s.head()

#%% Test unmask for dataframe

# Update the values in the masked series to something else, here 10
df_mask.iloc[:] = 10

# unmask and update values of original df
print(pd.read_csv(original_df).iloc[:,1].value_counts().head(2))
updated_df = tr.unmask_df(original_df, df_mask)
print(updated_df.iloc[:,1].value_counts().head(2))

print(pd.read_csv(original_df).shape)
print(df_mask.shape)
print(updated_df.shape)
updated_df.head()

#%% Test mask as long format dataframe
long_mask.iloc[:] = 10

# unmask and update values of original df
print(long_mask.iloc[:,1].value_counts().head(2))
updated_df = tr.unmask_df(my_df, long_mask)
print(updated_df.iloc[:,1].value_counts().head(2))

print(original_series.shape)
print(series_mask.shape)
print(updated_s.shape)
updated_df.head()