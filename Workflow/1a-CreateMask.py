# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:33:32 2018

@author: mmann
"""

#%%

from tsraster.prep import sRead as tr

#%%

# rasterize the state polygon to match inputs 
tr.poly_rasterizer(poly = 'F:/Boundary/StatePoly.shp', raster_ex = r'F:/5year/aet/aet-201201.tif',
                raster_path_prefix = r'F:/Boundary/StatePoly_buf',buffer_poly_cells=10)


#%%  CHECK MASK
raster_mask = u"F:/Boundary/StatePoly_buf.tif"
raster_input_ex = u"F:/5year/aet/aet-201201.tif"

tr.check_mask(raster_mask,raster_input_ex)



#%% Mask a dataframe

raster_mask = u"F:/Boundary/StatePoly_buf.tif"
original_df = u"F:/5year/aet_features/extracted_features.csv"

df_mask  = tr.mask_df(raster_mask, original_df)



#%% Mask a series 

original_series  = tr.targetData(raster_mask)

series_mask = tr.mask_df(raster_mask, original_series)


#%% UN-Mask a dataframe or series
import pandas as pd
 
def unmask_df(original_df, mask_df_output):
    '''
    mask_df: reads in raster mask and subsets df by mask index
         raster_mask - tif containing (0,1) mask 
         original_df - a pandas dataframe or series to mask can be path to csv or pandas series or dataframe
    '''
    
         # check if polygon is already geopandas dataframe if so, don't read again
    if not(isinstance(original_df, pd.core.series.Series)) and \
            not(isinstance(original_df, pd.core.frame.DataFrame)): 
        original_df = pd.read_csv(original_df)
    else:
        original_df = original_df
        
    # replace values based on masked values
    original_df.update(mask_df_output)

    return original_df


#%% Test unmask for series
# Update the values in the masked dataset to something else, here 10
series_mask.iloc[:] = 10

# unmask and update values of original df
print(original_series.iloc[:].value_counts().head(4))
updated_s = unmask_df(original_series, series_mask)
print(updated_s.iloc[:].value_counts().head(4))

#%% Test unmask for dataframe

# Update the values in the masked series to something else, here 10
df_mask.iloc[:] = 10

# unmask and update values of original df
print(pd.read_csv(original_df).iloc[:,1].value_counts().head(2))
updated_df = unmask_df(original_df, df_mask)
print(updated_df.iloc[:,1].value_counts().head(2))

