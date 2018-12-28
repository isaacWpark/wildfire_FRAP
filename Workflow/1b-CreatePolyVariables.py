# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:35:27 2018

@author: mmann
"""


import tsraster.prep as tr
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

#%%

# rasterize the state polygon to match inputs 
poly = 'F:/Boundary/Jepson.shp' 
raster_ex = r'F:/5year/aet/aet-201201.tif'
raster_path_prefix = None
buffer_poly_cells=0
field_name = 'JEPSON_ID'
nodata=-9999
plot_output = True 

a = tr.poly_to_series(poly,raster_ex, field_name, nodata=-9999, plot_output=True)
a.describe()


#%%

#def poly_to_series(poly,raster_ex, field_name,  nodata=-9999, plot_output=True):
    
'''
Rasterizes polygons by assigning a value 1.
It can also add a buffer at a distance that is multiples of the example raster resolution

:param poly: polygon to to convert to raster
:param raster_ex: example tiff
:param raster_path_prefix: directory path to the output file example: 'F:/Boundary/StatePoly_buf'
:param nodata: (int or float, optional) – Used as fill value for all areas not covered by input geometries.
:param nodata: (True False, optional) – Plot rasterized polygon data? 

:return: a GeoTiff raster
'''

# check if polygon is already geopandas dataframe if so, don't read again
if ('poly' in locals()):
    if not(isinstance(poly, gpd.geodataframe.GeoDataFrame)):
        poly = gpd.read_file(poly)
else:
    poly = poly


# get example metadata
with rasterio.open(raster_ex) as src:
    array = src.read()
    profile = src.profile
    profile.update(dtype=rasterio.float32, count=1, compress='lzw',nodata=nodata)
    out_arr = src.read(1) # get data from first band, this gets updated in write
    out_arr.fill(nodata) #set all values of raster to zero


# reproject polygon to match crs of raster
poly = poly.to_crs(src.crs)


# generator of geom, value pairs to use in rasterizing
shapes = ((geom,value) for geom, value in zip(poly.geometry, poly[field_name]))

#rasterize shapes
burned_value = features.rasterize(shapes=shapes, fill=nodata, out=out_arr, transform=src.transform)


#%%
if plot_output == True:
    import matplotlib.pyplot as plt 
    plt_burned_value = burned_value.copy()
    plt_burned_value[plt_burned_value==nodata] = np.NaN
    plt.imshow(plt_burned_value)
    plt.set_cmap("Reds")
    plt.colorbar( )
    plt.show()

# convert to array 
rows, cols = burned_value.shape
data = burned_value.reshape(rows*cols, 1)

# create index
index = pd.RangeIndex(start=0, stop=len(data), step=1) 

# create wide df with images as columns
df = pd.DataFrame(data=data[:,:],
                  index=index, 
                  dtype=np.float32, 
                  columns=[field_name])
#%%
return burned_value

#%%































