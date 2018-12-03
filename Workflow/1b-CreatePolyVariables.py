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
buffer_poly_cells=10
field_name = 'JEPSON_ID'


#%%

#def poly_to_series(poly,raster_ex, field_name, buffer_poly_cells=0):
'''
Rasterizes polygons by assigning a value 1.
It can also add a buffer at a distance that is multiples of the example raster resolution

:param poly: polygon to to convert to raster
:param raster_ex: example tiff
:param raster_path_prefix: directory path to the output file example: 'F:/Boundary/StatePoly_buf'
:param buffer_poly_cells: buffer size in cell count example: 1 = buffer by one cell
:return: a GeoTiff raster
'''

# check if polygon is already geopandas dataframe if so, don't read again
if ('poly' in locals()):
    if not(isinstance(poly, gpd.geodataframe.GeoDataFrame)):
        poly = gpd.read_file(poly)
else:
    poly = poly

print(poly.head(3))

# get example metadata
with rasterio.open(raster_ex) as src:
    array = src.read()
    profile = src.profile
    profile.update(dtype=rasterio.float32, count=1, compress='lzw',nodata=-9999)
    out_arr = src.read(1) # get data from first band, this gets updated in write
    out_arr.fill(0) #set all values of raster to zero


# reproject polygon to match crs of raster
poly = poly.to_crs(src.crs)

# buffer polygon to avoid edge effects
if buffer_poly_cells != 0:
    poly['geometry'] = poly.buffer(buffer_poly_cells*src.res[0] ) # this creates an empty polygon geoseries

#%%
#if raster_path_prefix is not None:
#    # Write to tif, using the same profile as the source
#    with rasterio.open(raster_path_prefix+'.tif', 'w', **profile) as dst:
#        # generator of geom, value pairs to use in rasterizing
#        shapes = ((geom,value) for geom, value in zip(poly.geometry, poly[field_name]))    
#        #rasterize shapes
#        burned_value = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=dst.transform)        
#        dst.write(burned_value,1)
#        
#else:
# generator of geom, value pairs to use in rasterizing
shapes = ((geom,value) for geom, value in zip(poly.geometry, poly[field_name]))

#rasterize shapes
burned_value = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=src.transform)

 #   return burned_value 

#%%
import matplotlib.pyplot as plt 

plt.imshow(burned_value)
plt.set_cmap("Reds")
plt.colorbar( )
plt.show()






























