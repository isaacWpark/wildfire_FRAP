# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:33:32 2018

@author: mmann
"""



#%%
import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np   

def poly_rasterizer(poly,raster_ex, raster_path_prefix, buffer_poly_cells=0):
    '''
    :poly_rasterizer: Function rasterizes polygons assigning the value 1, it \\
    can also add a buffer at a distance that is multiples of the example raster resolution
    :param raster_path_prefix: full path and prefix for raster name 
    :param buffer_poly_cells: int specifying number of cells to buffer polygon with, 0 for no buffer
    :return: raster
     '''
     
    # check if polygon is already geopandas dataframe if so, don't read again
    if ('poly' in locals()):
        if not(isinstance(poly, gpd.geodataframe.GeoDataFrame)): 
            poly = gpd.read_file(poly)
    else:
        poly = poly
    
    # create column of ones to rasterize for presence (1) of fire
    poly['ONES'] = 1
       
    # get example metadata
    with rasterio.open(raster_ex) as src:
        array = src.read()
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw',nodata=0)
        out_arr = src.read(1) # get data from first band, this gets updated in write
        out_arr.fill(0) #set all values of raster to zero 
        
    # reproject polygon to match crs of raster
    poly = poly.to_crs(src.crs)
    
    # buffer polygon to avoid edge effects 
    if buffer_poly_cells != 0:
        poly['geometry'] = poly.buffer(buffer_poly_cells*src.res[0] ) # this creates an empty polygon geoseries
        
    # Write to tif, using the same profile as the source
    with rasterio.open(raster_path_prefix+'.tif', 'w', **profile) as dst:

        # generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(poly.geometry, poly.ONES))

        #rasterize shapes 
        burned_value = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=dst.transform)
        dst.write(burned_value,1)
        
        
     
        
    
        
#%% 


poly_rasterizer(poly = 'F:/Boundary/StatePoly.shp', raster_ex = r'F:/3month/aet-198401.tif',
                raster_path_prefix = r'F:/Boundary/StatePoly_buf',buffer_poly_cells=10)