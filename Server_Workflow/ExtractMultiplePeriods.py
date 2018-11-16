# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 09:50:54 2018

@author: mmann
"""

from tsraster.calculate import calculateFeatures2
 
fc_parameters = {
	    "mean": None,
	    "maximum": None,
	    "median":None,
	    "minimum":None,
        "sum_values":None,
	    "agg_linear_trend": [{"attr": 'slope', "chunk_len": 6, "f_agg": "min"},
                              {"attr": 'slope', "chunk_len": 6, "f_agg": "max"}],
	    "last_location_of_maximum":None,
	    "last_location_of_maximum":None,
	    "last_location_of_minimum":None,
	    "longest_strike_above_mean":None,
	    "longest_strike_below_mean":None,
	    "count_above_mean":None,
	    "count_below_mean":None,
	    #"mean_abs_change":None,
	    "mean_change":None,
	    "number_cwt_peaks":[{"n": 6},{"n": 12}],
	    "quantile":[{"q": 0.15},{"q": 0.05},{"q": 0.85},{"q": 0.95}],
	    "ratio_beyond_r_sigma":[{"r": 2},{"r": 3}], #Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.
	    "skewness":None,
	}


from pathlib import Path 
import os
p = Path('/home/mmann1123/wildfire_FRAP/Data/Actual/Climate/')
folders = [str(x) for x in p.glob('*/*') if x.is_dir() and '_features' not in str(x) and 'Fire' not in str(x) ]
folders

for folder in folders:
    if not os.listdir(folder):
        print("Directory is empty")
    else:    
        print("Directory is not empty")
        print(folder)
        ts_features = calculateFeatures2(path=folder, 
					parameters=fc_parameters, 
					mask="/home/mmann1123/wildfire_FRAP/Data/Actual/Boundary/StatePoly_buf.tif", 
					reset_df=True, 
					tiff_output=True, 
					missing_value =-9999,
					workers=8)
        print(ts_features.describe())
        del ts_features
 
