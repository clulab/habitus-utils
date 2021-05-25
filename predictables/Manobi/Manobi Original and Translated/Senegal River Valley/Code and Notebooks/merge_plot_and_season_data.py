#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:18:40 2021

@author: prcohen


This file does only one thing: It moves plot data from the plot file into the 
season file.  It's only a few bits of information.

"""

import sys
sys.path.append('/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/Manobi farmer')
import numpy as np
import pandas as pd


filepath = "/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/Manobi Data/Original Steven 2017b/"
village = pd.read_excel(filepath+"translated_2017b/village_2017b_translated.xls")
farmer  = pd.read_excel(filepath+"translated_2017b/farmer_2017b_translated.xls")
plot    = pd.read_excel(filepath+"translated_2017b/plot_2017b_translated.xls")

# Use Allegra's cleaned seasdson data rather than the original (horrible) 
# agSeason_2017b_translated.xls

filepath = "/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/Manobi Data/Original Steven 2017b/"
season  = pd.read_excel(filepath+"processed/season_StLouis_2016_english.xls")

def strip_spaces (df):
    # strip leading and trailing spaces in all string-valued column names and values
    df.columns = [x.strip() for x in df.columns]
    return df.applymap(lambda x: x.strip() if type(x) == str else x)

plot = strip_spaces(plot)

season_temp = season.set_index('plot_ID',drop=False)
df = plot.set_index('X.ID',drop=False)
df = df.join(season_temp)

# 379 of 3462 plots have no season data
df.drop(df[df['plot_ID'].isna()].index, inplace=True)

# rename the columns from the plot data and two columns from season data
df.rename({'X.ID':'plot_ID_from_plot',
           'plot_ID' : 'plot_ID_from_season',
           'ID' : 'season_ID',              # from season data
           'Parent.ID' : 'farm_ID',         # parent of plot = farm
           'Date' : 'collection_date_from_plot', 
           'collection_date' : 'collection_date_from_season',
           'numPlots' : 'plot_number',
           'ownershipInfo' : 'plot_owner',
           'plotArea' : 'plot_area',
           'DElimitation.parcelle' : 'plot_boundary',
           'CentroOde.parcelle.' : 'plot_lat_long',
           'Soil type' : 'plot_soil_type',
           'Soil color' : 'plot_soil_color',
           'soilQuality' : 'plot_soil_quality',
           'Topography' : 'plot_topography'
           },
          axis = 1, inplace=True
          )
           
 