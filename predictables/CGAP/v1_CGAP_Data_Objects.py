#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:49:27 2021

@author: prcohen
"""

import sys
import os, json
import copy
from types import SimpleNamespace
import os, json

import numpy as np
import pandas as pd

sys.path.append('/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/CGAP/Data/Data Objects/')

from CGAP_JSON_Encoders_Decoders import Question_Encoder as QE
from CGAP_JSON_Encoders_Decoders import Question_Decoder, CGAP_Encoded, CGAP_Decoded, Country_Decoded





# Encoding and decoding questions to and from JSON

 

filepath = '/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/CGAP/Data/Processed SRS and MRS Data/'

def read_csv (path):
    return pd.read_csv(filepath+path, low_memory=False)

#_____________________________________________________________________________
countries = ['bgd','cdi','moz','nga','tan','uga']



#mithun wrapping a function around the data reading and conversion
def get_data():
    # dict holds survey data; e.g., data_dict['bgd_sr'].shape
    data_dict = {}

    # read in the single-respondent and reduced multi-respondent data
    for country in countries:
        for survey in ['sr', 'rr']:
            data_dict[country + '_' + survey] = read_csv(country + '_' + survey + '_clean.csv')

    # read in the household survey data
    filepath = '/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/CGAP/Data/Processed Household Data/Level 1/'

    for country in countries:
        data_dict[country + '_hh'] = read_csv(country + '_level1.csv')

    # Hack so I don't have to change all the code below to add data_dict as an additional parameter
    def Question_Encoder(**kwargs):
        return QE(df=data_dict, **kwargs)

    # _____________________________________________________________________________
    # Really important:  HHID must be the index if we're to re-assemble dataframes
    # from individual columns

    for country in countries:
        for survey in ['sr', 'rr', 'hh']:
            data_dict[country + '_' + survey].set_index('HHID', drop=False, inplace=True)


# dict holds survey data; e.g., data_dict['bgd_sr'].shape
data_dict = {}

# read in the single-respondent and reduced multi-respondent data
for country in countries:
    for survey in ['sr','rr']:
        data_dict[country+'_'+survey] = read_csv(country+'_'+survey+'_clean.csv')

# read in the household survey data
filepath = '/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/CGAP/Data/Processed Household Data/Level 1/'
 
for country in countries:
    data_dict[country+'_hh'] = read_csv(country+'_level1.csv')

# Hack so I don't have to change all the code below to add data_dict as an additional parameter
def Question_Encoder (**kwargs):
   return QE(df = data_dict, **kwargs)

#_____________________________________________________________________________
# Really important:  HHID must be the index if we're to re-assemble dataframes
# from individual columns

for country in countries:
    for survey in ['sr','rr', 'hh']:
        data_dict[country+'_'+survey].set_index('HHID',drop=False,inplace=True)


def get_var(country,var):
    try:
        return data_dict[country][var]
    except:
        print(f"Can't find {var}. Searching for similar...")
        for c in data_dict[country].columns:
            if var in c:
                print (c)
    
   
#_____________________________________________________________________________
# Objects that hold all the encoded questions and decoded questions    
cgap = CGAP_Encoded()
Data = CGAP_Decoded()

#_____________________________________________________________________________

## functions to streamline the workflow, below

def change_dict (d, to_delete, to_add):
    for item in to_delete: 
        del d[item]
    for item in to_add:
        d.update(item)
        

def encode_and_add (Q):
    for i in range(len(Q)) : 
        Q[i].make_columns()
        cgap.add_encoded(Q[i].encode())


"""
This example illustrates the basic workflow.  First, as all countries are generally
similar, build six question encoders that are identical but for the country. 
Second, do surgery on the answers dict or the columns_dict. Third, make the
data columns. Finally, add the encoded json strings to the CGAP_Encoded object
"""

Q = [Question_Encoder (
        qtype = 'single',
        text =  'What is the form of ownership of your land?',
        label = 'A1',
        answers = {'lease_certificate' : 1, 'customary_law' : 2, 'communal' : 3,
                   'state_ownership' : 4, 'other' : 5},
        country = country,
        survey = 'rr'
        ) 
    for country in countries]

# bgd and cdi have extra answers
change_dict(Q[0].answers,['other'],[{'Kott' : 5, 'other': 6}])
change_dict(Q[1].answers,['other'],[{'sharecropping' : 5, 'other': 6}]) 

# run the encoders to encode and add data, then add the encoded
# questions to cgap, which is a CGAP_Encoded() object
    
encode_and_add(Q)

#____________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'How many hectares of agricultural land do you own?',
        label = 'A2',
        country = country,
        survey = 'rr'
        ) 
    for country in countries]

# Identical across countries, no surgery

encode_and_add(Q)
    
#____________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'How many hectares of agricultural land do rent, borrow or have the right to use?',
        label = 'A3',
        country = country,
        survey = 'rr'
        ) 
    for country in countries]

# Identical across countries, no surgery

encode_and_add(Q)
    
#____________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Do you consider your farm to be a business?',
        label = 'A4',
        answers = {'yes' : 1, 'no' : 2},
        country = country,
        survey = 'rr'
        ) 
    for country in countries]

# Identical across countries, no surgery

encode_and_add(Q)

#_____________________________________________________________________________

bgd_crops = {'Rice': 1, 'Wheat': 2, 'Mango': 3, 'Jute': 4, 'Maize': 5, 'Tea': 6, 'Pulses': 7, 
             'Sugarcane': 8, 'Tobacco': 9, 'Chilies': 10, 'Onions': 11, 'Garlic': 12, 
             'Potato': 13, 'Rapeseed': 14, 'Mustard_seed': 15, 'Coconut': 16, 'Eggplant': 17, 
             'Radish': 18, 'Tomatoes': 19, 'Cauliflower': 20, 'Cabbage': 21, 'Pumpkin': 22, 
             'Banana': 23, 'Jackfruit': 24, 'Pineapple': 25, 'Guava': 26, 'Sesame': 27, 
             'Other_1': 28, 'Other_2': 29, 'Other_3': 30, 'No_crop': 31}

cdi_crops = {'Rice': 1, 'Maize': 2, 'Cocoa': 3, 'Coffee': 4, 'Sweet_potato': 5, 'Cassava': 6, 
             'Cashew': 7, 'Plantain': 8, 'Groundnuts': 9, 'Millet': 10, 'Palm_oil': 11, 
             'Hevea': 12, 'Okra': 13, 'Chilies': 14, 'Onions': 15, 'Eggplant': 16, 
             'Tomatoes': 17, 'Cabbage': 18, 'Pumpkin': 19, 'Cucumber': 20, 'Salad': 21, 
             'Sesame': 22, 'Sugarcane': 23, 'Mango': 24, 'Papaya': 25, 'Orange': 26, 
             'Coconut': 27, 'Banana': 28, 'Pineapple': 29, 'Guava': 30, 'Cotton': 31, 
             'Yams': 32, 'Other_1': 33, 'Other_2': 34, 'Other_3': 35, 'No_crop': 36}

moz_crops = {'Maize': 1, 'Beans': 2, 'Sweet_potato': 3, 'Sorghum': 4, 'Rice': 5, 
             'Groundnuts': 6, 'Cowpea': 7, 'Millet': 8, 'Cassava': 9, 'Potato': 10, 
             'Pigeon_pea': 11, 'Banana': 12, 'Coconut': 13, 'Cotton': 14, 'Sesame': 15, 
             'Mango': 16, 'Cashew': 17, 'Sugarcane': 18, 'Tobacco': 19, 'Tea': 20, 
             'Avocado': 21, 'Cocoa': 22, 'Sisal': 23, 'Cloves': 24, 'Coffee': 25, 
             'Sunflower': 26, 'Tomatoes': 27, 'Onions': 28, 'Other_1': 29, 
             'Other_2': 30, 'Other_3': 31, 'No_crop': 32}
       
nga_crops = {'Wheat': 1, 'Rice': 2, 'Maize': 3, 'Millet': 4, 'Sorghum': 5, 
             'Fonio': 6, 'Potato': 7, 'Sweet_potato': 8, 'Cassava': 9, 'Taro': 10, 
             'Yams': 11, 'Sugarcane': 12, 'Cowpea': 13, 'Pulses': 14, 'Beans': 15, 
             'Soybeans': 16, 'Groundnuts': 17, 'Coconut': 18, 'Palm_oil': 19, 
             'Karite_Shea': 20, 'Sesame': 21, 'Melonseed': 22, 'Seed_cotton': 23, 
             'Tomatoes': 24, 'Cabbage': 25, 'Onions': 26, 'Cashew': 27, 'Banana': 28, 
             'Cotton': 29, 'Tobacco': 30, 'Pyrethrum': 31, 'Coffee': 32, 'Orange': 33, 'No_crop': 34}

tan_crops = {'Maize': 1, 'Rice': 2, 'Sorghum': 3, 'Millet': 5, 'Cassava': 6, 
             'Sweet_potato': 7, 'Potato': 8, 'Beans': 9, 'Cowpea': 10, 'Pigeon_pea': 11, 
             'Sunflower': 12, 'Sesame': 13, 'Groundnuts': 14, 'Tomatoes': 15, 
             'Cabbage': 16, 'Onions': 17, 'Amaranth': 18, 'Cashew': 19, 'Banana': 20, 
             'Cotton': 21, 'Tobacco': 22, 'Pyrethrum': 23, 'Coffee': 24, 'Coconut': 25, 
             'Orange': 26, 'Sugarcane': 27, 'Palm_oil': 28, 'Other_1': 29, 
             'Other_2': 30, 'Other_3': 31, 'No_crop': 32}

uga_crops = {'Maize': 1, 'Beans': 2, 'Sweet_potato': 3, 'Sorghum': 4, 'Rice': 5, 
             'Groundnuts': 6, 'Cowpea': 7, 'Millet': 8, 'Cassava': 9, 'Potato': 10, 
             'Pigeon_pea': 11, 'Banana': 12, 'Cotton': 13, 'Sesame': 21, 
             'Sugarcane': 15, 'Tobacco': 16, 'Tea': 17, 'Cocoa': 18, 'Coffee': 19, 
             'Field_pea': 20, 'Soybeans': 22, 'Other_1': 23, 'Other_2': 24, 
             'Other_3': 25, 'No_crop': 26}

# Now we're going to loop over all the multi-answer questions about crops:

labels = ['A5','A7','A9','A25','A36']
texts = [
    'Which of the following crops do you grow?',
    'Which of the following crops do you grow that you consume at home?',
    'Do you buy any of the following crops?',
    'Which of the following crops that you grow do you sell?', 
    'Which of the following crops that you grow do you trade?'
        ]
    

for label,text in zip(labels,texts):
    print (label)
    print(text)
    Q = [Question_Encoder (
        qtype = 'multi',
        text =  text,
        label = label,
        answers = {1 : 'yes', 2 : 'no'},
        column_dict = eval(country+'_crops'),
        country = country,
        survey = 'rr'
        )
        for country in countries]
    
    encode_and_add(Q)

#_____________________________________________________________________________

# A53 is another crop question, though it's in the single-respondent data
# so we'll use the same strategy 
Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Which crops do you normally store?' ,
        label = 'A53',
        answers = {1 : 'yes', 2 : 'no'},
        column_dict = eval(country+'_crops'),
        country = country,
        survey = 'sr'
        )
        for country in countries]
    
encode_and_add(Q)
 
#_____________________________________________________________________________            
"""
That's part of the problem solved.  The other part is that single-answer questions about
crops are coded differently in different countries.  For example, A6 asks which is the
most important crop to the household, but 1 means maize in some countries and rice in others.
The solution is to build a master dict for crops and use it to translate the numeric codes
in each country into common codes.
"""

master_crop_dict = {
    'Rice': 1, 'Wheat': 2, 'Maize': 3, 'Millet': 4, 'Sorghum': 5, 'Fonio': 6, 
    'Cassava': 7, 'Groundnuts': 8, 'Beans': 9, 'Cowpea': 10, 'Field_pea': 11, 
    'Pigeon_pea': 12, 'Pulses': 13, 'Soybeans': 14, 'Plantain': 15, 'Potato': 16, 
    'Sweet_potato': 17, 'Taro': 18, 'Yams': 19, 'Avocado': 20, 'Banana': 21, 
    'Cabbage': 22, 'Cauliflower': 23, 'Chilies': 24, 'Cucumber': 25, 'Eggplant': 26, 
    'Garlic': 27, 'Guava': 28, 'Jackfruit': 28, 'Mango': 29, 'Melonseed': 30, 'Okra': 31, 
    'Onions': 32, 'Pineapple': 32, 'Orange': 33, 'Pumpkin': 34, 'Radish': 35, 
    'Rapeseed': 36, 'Salad': 37, 'Sunflower': 38, 'Tomatoes': 39, 'Papaya': 40, 
    'Cashew': 41, 'Cloves': 42, 'Cocoa': 43, 'Coconut': 44, 'Coffee': 45, 'Cotton': 46, 
    'Hevea': 47, 'Jute': 48, 'Karite_Shea': 49, 'Mustard_seed': 50, 'Palm_oil': 51, 
    'Amaranth': 52, 'Pyrethrum': 53, 'Seed_cotton': 54, 'Sesame': 55, 'Sisal': 56, 
    'Sugarcane': 57, 'Tea': 58, 'Tobacco': 59, 
    'Other_1': 60, 'Other_2': 61, 'Other_3': 62, 'No_crop': 63}


# We have to fix Mozambique and Uganda, which code crops as strings.   
# Unfortunately, the terms used to denote crops in moz and uga aren't 
# all exact matches to the terms in master_dict

master_crop_dict.update({
    'Bananas': master_crop_dict['Banana'],
    'Pigeon pea': master_crop_dict['Pigeon_pea'],
    'Sugar cane': master_crop_dict['Sugarcane'],
    'Simsim': master_crop_dict['Sesame'],
    'Sweet potatoes': master_crop_dict['Sweet_potato'],
    'Irish potatoes': master_crop_dict['Potato'],
    'Soya beans': master_crop_dict['Soybeans'],
    'Field peas': master_crop_dict['Field_pea'],
    'Other 1': master_crop_dict['Other_1'],
    'Other 2': master_crop_dict['Other_2'],
    'Other 3': master_crop_dict['Other_3'],
    'Other1': master_crop_dict['Other_1'],
    'Other2': master_crop_dict['Other_2'],
    'Other3': master_crop_dict['Other_3'],
    'None': master_crop_dict['No_crop'],
}) 


# now we can recode crop names as numbers in moz and uga
for var in ['A6','A8','A26','A37']:
    data_dict['moz_rr'][var].replace(master_crop_dict,inplace=True)
    data_dict['uga_rr'][var].replace(master_crop_dict,inplace=True)


# for the other countries, we build a dict for each that maps its own
# crop number to the master_crop_dict number for that crop

for var in ['A6','A8','A26','A37']:
    data_dict['bgd_rr'][var].replace({v : master_crop_dict[k] for k,v in bgd_crops.items()}
                        ,inplace=True)
    data_dict['cdi_rr'][var].replace({v : master_crop_dict[k] for k,v in cdi_crops.items()}
                        ,inplace=True)
    data_dict['nga_rr'][var].replace({v : master_crop_dict[k] for k,v in nga_crops.items()}
                        ,inplace=True)
    data_dict['tan_rr'][var].replace({v : master_crop_dict[k] for k,v in tan_crops.items()}
                        ,inplace=True)


# finally, we can encode the questions

labels = ['A6','A8','A26','A37']
texts = [
    'Which of the following crops is most important to you?',
    'Which of the following crops do you consume the most at home?',
    'Which of the following crops that you make most money from?', 
    'Which of the following crops do you trade? the most'
        ]

for label,text in zip(labels,texts):
    Q = [Question_Encoder (
        qtype = 'single',
        text =  text,
        label = label,
        answers = {k : master_crop_dict.get(k) for k in eval(country+'_crops').keys()},
        country = country,
        survey = 'rr'
        )
        for country in countries]
    
    encode_and_add(Q)
    

#_____________________________________________________________________________
# Now, onto livestock

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Do you have any livestock herds, other farm animals, or poultry?',
        label = 'A10',
        answers = {'yes' : 1, 'no' : 2},
        country = country,
        survey = 'rr'
        ) 
    for country in countries]

# Identical across countries, no surgery

encode_and_add(Q)
    

#_____________________________________________________________________________
        
# Now we have to turn the same cartwheels for livestock as we did for crops

bgd_livestock = {
    'Cattle_beef': 1, 'Cattle_dairy': 2, 'Buffalo': 3, 'Goat_meat': 4, 'Goat_dairy': 5, 
    'Sheep': 6, 'Chicken_broiler': 7, 'Chicken_layer': 8, 'Duck': 9, 'Pigeon': 10, 
    'Fish': 11, 'Bees': 12, 'Other': 13, 'No_livestock': 14}

cdi_livestock = {
    'Cattle_beef': 1, 'Cattle_dairy': 2, 'Buffalo': 3, 'Goat_meat': 4, 'Goat_dairy': 5, 
    'Sheep': 6, 'Chicken_broiler': 7, 'Chicken_layer': 8, 'Pig': 9, 'Duck': 10, 
    'Pigeon': 11, 'Fish': 12, 'Bees': 13, 'Other': 14, 'No_livestock': 15}

moz_livestock = {
    'Cattle_beef': 1, 'Cattle_dairy': 2, 'Cattle_ind': 3, 'Sheep': 4, 'Duck': 5, 
    'Pig': 6, 'Goat_meat': 7, 'Goat_dairy': 8, 'Chicken_broiler': 9, 
    'Chicken_layer': 10, 'Fish': 11, 'Bees': 12, 'Other': 13, 'No_livestock': 14}
    
nga_livestock = {
    'Cattle_ind': 1, 'Cattle_dairy': 2, 'Cattle_beef': 3, 'Goat_ind': 4, 
    'Goat_dairy': 5, 'Goat_meat': 6, 'Sheep': 7, 'Pig': 8, 'Chicken_layer': 9, 
    'Chicken_broiler': 10, 'Fish': 11, 'Bees': 12, 'Camel': 13, 'Donkey': 14, 
    'Horse': 15, 'Dog': 16, 'Other': 17, 'No_livestock': 18}

tan_livestock = {
    'Cattle_ind': 1, 'Cattle_dairy': 2, 'Cattle_beef': 3, 'Goat_ind': 4, 
    'Goat_dairy': 5, 'Goat_meat': 6, 'Sheep': 7, 'Pig': 8, 'Chicken_layer': 9, 
    'Chicken_broiler': 10, 'Fish': 11, 'Bees': 12, 'Other': 13, 'No_livestock': 14}

uga_livestock = {
    'Cattle_beef': 1, 'Cattle_dairy': 2, 'Cattle_ind': 3, 'Sheep': 4, 'Duck': 5, 
    'Turkey': 6, 'Pig': 7, 'Goat_meat': 8, 'Goat_dairy': 9, 'Chicken_broiler': 10, 
    'Chicken_layer': 11, 'Fish': 12, 'Bees': 13, 'Other': 14, 'No_livestock': 15}
   
#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'How many of each of the following do you rear?',
        label = 'A11',
        column_dict = eval(country+'_livestock'),
        country = country,
        survey = 'rr'
        )
        for country in countries]
    
encode_and_add(Q)
        

labels = ['A12','A14']
texts = [
    'Which of the following do you rear AND get an income from?',
    'Which of the following livestock or byproducts do you rear to consume at home?' ,  
        ]
    

for label,text in zip(labels,texts):
    print (label)
    print(text)
    Q = [Question_Encoder (
        qtype = 'multi',
        text =  text,
        label = label,
        answers = {1 : 'yes', 2 : 'no'},
        column_dict = eval(country+'_livestock'),
        country = country,
        survey = 'rr'
        )
        for country in countries]
    
    encode_and_add(Q)

#_____________________________________________________________________________   
# The only single-answer question in which the answers denote kinds of livestock is A13:
    
master_livestock_dict = {
    'Cattle_beef': 1, 'Cattle_dairy': 2, 'Cattle_ind': 3, 'Buffalo': 4, 
    'Goat_meat': 5, 'Goat_dairy': 5, 'Goat_ind' : 6, 'Sheep': 7, 'Pig' : 8,
    'Chicken_broiler': 9, 'Chicken_layer': 10, 'Duck': 11, 'Pigeon': 12,
    'Turkey': 13, 'Camel': 14, 'Donkey': 15, 'Horse': 16, 'Dog': 17,
    'Fish': 18, 'Bees': 19, 'Other': 20, 'No_livestock': 21
    }
    
# for all the countries, we build a dict for each that maps its own
# livestock number to the master_livestock_dict number for that crop. 

for c in countries:
    data_dict[c+'_rr']['A13'].replace({v : master_livestock_dict[k]
                                  for k,v in eval(c+'_livestock').items()},
                                 inplace=True)

    
Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Who do you normally purchase your main agricultural and livestock inputs (such as seeds, fertilizer, or pesticide) from?',
        label = 'A15',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'cooperative':1, 'wholesaler':2,' processor':3, 'retailer':4,
                       'government':5, 'middleman':6, 'other':7, 'no_purchase':8, 'DK' : 98},
        country = country,
        survey = 'rr'
        )
        for country in countries]

# moz skips code 5, 8 and 9
Q[2].column_dict = {
    'cooperative':1, 'wholesaler':2,' processor':3, 'retailer':4,
    'government':6, 'middleman':7, 'other':10, 'no_purchase':11, 'DK' : 98}

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'How do you usually pay your suppliers?',
        label = 'A17',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'cash':1, 'cheque':2,'pay_cash_bank':3, 'electronic':4,
                       'mobile_banking':5, 'in_kind':6, 'prepaid_card':7, 
                       'other':8, 'do_not_buy' : 9, 'DK' : 98},
        country = country,
        survey = 'rr'
        )
        for country in countries]

# cdi has additional options
change_dict (Q[1].column_dict, 
             ['other','do_not_buy'], 
             [{'direct_from_loan' : 8, 'deduct_at_harvest' : 9, 'other' : 10, 'do_not_buy' : 11}]
             )
    
encode_and_add(Q)
#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Do your suppliers give you the option to pay them later (with credit) or do you have to pay them immediately?',
        label = 'A18',
        answers = {'pay_later' : 1, 'pay_immediately' : 2},
        country = country,
        survey = 'rr'
        ) 
    for country in countries]

# Identical across countries, no surgery

encode_and_add(Q)
    
#_____________________________________________________________________________

# I re-ordered the responses from "bad to good" when I cleaned the data 
Q = [Question_Encoder (
        qtype = 'single',
        text =  'Which of the following statements best describe your water situation?',
        label = 'A22',
        answers = {'would_use_more' : 4, 'sufficient' : 3, 'intermittent_but_ok' : 
                   2, 'intermittent_not_ok' : 1},
        country = country,
        survey = 'rr'
        ) 
    for country in countries]

# Identical across countries, no surgery

encode_and_add(Q)
    
#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'multi',
        text =  'For managing the land and livestock, what types of labor do you use?',
        label = 'A23',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'friends_neighbors':1, 'hired_extended_period':2,
                       'family':3, 'day_labor':4, 'other':5, 'no_labor' : 6},
        country = country,
        survey = 'rr',
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'What do you use the labor for?',
        label = 'A24',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'ploughing':1, 'planting':2,'weeding':3, 
                       'harvesting':4, 'selling_crops' : 5, 'livestock_care' : 6,
                       'livestock_sale' : 7, 'other':8},
        country = country,
        survey = 'rr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Who do you sell your crops and livestock to?',
        label = 'A27',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'cooperative':1, 'wholesaler':2,'processor':3, 
                       'retailer':4, 'public' : 5, 'government' : 6,
                       'middleman' : 7, 'other':8, 'DK' : 98},
        country = country,
        survey = 'rr'
        )
        for country in countries]

change_dict(Q[1].column_dict,['other'],[{'agribusiness':8,'other':9}])
change_dict(Q[2].column_dict,['other'],[{'other':10}])
change_dict(Q[5].column_dict,['other'],[{'other':10}])


encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Where do you normally sell your crops and livestock?',
        label = 'A28',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'at_farm':1, 'in_village':2,'local_market':3, 
                       'regional_market':4, 'other' : 5},
        country = country,
        survey = 'rr'
        )
        for country in countries]

change_dict(Q[1].column_dict,['other'],[{'farm_side':5,'other':6}])

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Why do you sell your crops and livestock at this location?',
        label = 'A29',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'best_price':1, 'no_transport':2,'poor_roads':3, 
                       'unaware_of_prices':4, 'small_production' : 5, 
                       'other' : 6, 'DK' : 98},
        country = country,
        survey = 'rr'
        )
        for country in countries]

change_dict(Q[2].column_dict,['small_production'],[{'other':5}])
change_dict(Q[5].column_dict,['small_production'],[{'other':5}])

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'When you sell your crops and livestock, do you get the current market price?',
        label = 'A30',
        answers = {'yes' : 1, 'no' : 2},
        country = country,
        survey = 'rr'
        ) 
    for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________
# All countries except Mozambique code A31 as a single-answer question.
# moz codes it as multi-answer.  I'll have to fix this later

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Why do you not get the current market price?',
        label = 'A31',
        answers = {'few_customers':1, 'take_advantage':2,'high_commissions':3, 
                   'corruption':4, 'no_transport' : 5, 'poor_quality' : 6,
                   'dont_know_why' : 7, 'other' : 8, 'DK' : 98},
        country = country,
        survey = 'rr'
        )
        for country in ['bgd', 'cdi','nga', 'tan', 'uga']]

change_dict(Q[0].answers,['poor_quality','dont_know_why','other'],[{'other':6}])
change_dict(Q[3].answers,['dont_know_why','other'],[{'other':7}])

encode_and_add(Q)

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Why do you not get the current market price?',
        label = 'A31',
        answers = {'yes' : 1, 'no' : 2},
        column_dict = {
            'few_customers':1, 'take_advantage':2,'high_commissions':3, 
            'corruption':4, 'no_transport' : 5, 'poor_quality' : 6,
            'dont_know_why' : 7, 'other' : 8},
        country = country,
        survey = 'rr'
        )
        for country in ['moz']]
encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  ' Do you have a contract to sell any of your crops or livestock?',
        label = 'A32',
        answers = {'yes':1, 'no':2},
        country = country,
        survey = 'rr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

# This one is pretty messed up! 

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'How do you usually get paid for what you sell?',
        label = 'A33',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'cash':1, 'cheque':2,'electronic':3,'mobile_banking':4, 
                       'in_kind': 5, 'prepaid_card':6, 'other':7},
        country = country,
        survey = 'rr'
        )
        for country in countries]

# cdi has an extra option:
Q[1].column_dict = {'cash':1, 'cheque':2, 'in_bank_account': 3, 'electronic':4,
                    'mobile_banking':5, 'in_kind': 6, 'prepaid_card':7, 'other':8}

# moz, tan and uga number the answers differently (i.e, no column A33_3)
# tan's User Guide numbering is wrong; it has columns A33_1, A33_2, A33_4,
# A33_5, A33_6, A33_7, A33_8

for i in [2,4,5]:
    Q[i].column_dict = {'cash':1, 'cheque':2, 'electronic':4,'mobile_banking':5, 
                        'in_kind': 6, 'prepaid_card':7, 'other':8}

encode_and_add(Q)


#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'What challenges do you face in terms of getting your crops and livestock to your customers?',
        label = 'A35',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'distance':1, 'transport':2,'damage':3,'lack_storage':4, 
                       'lack_refrigeration': 5, 'unreliable_middlemen':6, 
                       'no_challenges' : 7, 'other':8},
        country = country,
        survey = 'rr'
        )
        for country in countries]


# moz and uga number the answers differently (i.e, 1,2,5,6,7,8,9,10).  However,
# uga column names are consistent with the numbering in other countries (except moz),
# so uga's User Guide numbering is wrong. moz, however, needs a column dict that
# matches its column numbers


Q[2].column_dict = {'distance':1, 'transport':2,'damage':5,'lack_storage':6, 
                       'lack_refrigeration': 7, 'unreliable_middlemen':8, 
                       'no_challenges' : 9, 'other':10}

encode_and_add(Q)


#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Do you generate income from the following sources?',
        label = 'H1',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'regular_job':1, 'occasional_job':2,'retail_business':3,'services_business':4,
                       'grant_pension': 5, 'family_friends':6, 'growing_crops' : 7, 
                       'rearing_livestock':8, 'other' : 9},
        country = country,
        survey = 'rr'
        )
        for country in countries]

encode_and_add(Q)


#_____________________________________________________________________________
labels = ['H2A','H2B','H3','H4','H5']
texts  = [
    'Which of these has been your main source of income in the last month?',
    'Which of these has been your main source of income in the last year?',
    'Which of the following income sources is most important to you?',
    'Which of the following income sources do you like getting the most?',
    'Which of the following income sources is most reliable for you?'
    ]

for text,label in zip(texts,labels):  
    Q = [Question_Encoder (
            qtype = 'single',
            text =  text,
            label = label,
            answers = {'regular_job':1, 'occasional_job':2,'retail_business':3,'services_business':4,
                       'grant_pension': 5, 'family_friends':6, 'growing_crops' : 7, 
                       'rearing_livestock':8, 'other' : 9},
            country = country,
            survey = 'rr'
            )
            for country in countries]
    
    encode_and_add(Q)
    
    

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'What is your primary job (i.e., the job where you spend most of your time)?',
        label = 'H6',
        answers = {'farmer':1, 'professional':2,'shop_owner':3,
                   'business_owner':4,'laborer': 5, 'other':6},
        country = country,
        survey = 'rr'
        )
        for country in countries]
    
encode_and_add(Q)
    

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'What are your secondary or side jobs?',
        label = 'H7',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'farmer':1, 'professional':2,'shop_owner':3,
                       'business_owner':4,'laborer': 5, 'other': 6,
                       'no_secondary_job' : 7},
        country = country,
        survey = 'rr'
        )
        for country in countries]

encode_and_add(Q)


#_____________________________________________________________________________
# Mozambique and Uganda treat H8 as a single-answer question when it clearly
# is multi-answer.  See also D17 from the Household survey, below.

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'How frequently do you receive your main source of income...?',
        label = 'H8',
        answers = {'daily' : 1, 'weekly': 2, 'monthly':3, 'annually' : 4, 
                   'after_a_time' : 5, 'according_to_harvest': 6, 'DK' : 7},
        column_dict = {'regular_job':1, 'occasional_job':2,'retail_business':3,'services_business':4,
                       'grant_pension': 5, 'family_friends':6, 'growing_crops' : 7, 
                       'rearing_livestock':8, 'other' : 9},
        country = country,
        survey = 'rr'
        )
        for country in ['bgd','cdi','nga','tan']]

encode_and_add(Q)


Q = [Question_Encoder (
        qtype = 'single',
        text =  'How frequently do you receive your main source of income...?',
        label = 'H8',
        answers = {'daily' : 1, 'weekly': 2, 'monthly':3, 'annually' : 4, 
                   'after_a_time' : 5, 'according_to_harvest': 6, 'DK' : 7},
        country = country,
        survey = 'rr'
        )
        for country in ['moz','uga']]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Are there any other ways that you get income?',
        label = 'H9',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'processor':1, 'seller':2,'services':3,'rent_out_land':4,
                       'other' : 5, 'no_other_way' : 6},
        country = country,
        survey = 'rr'
        )
        for country in countries]

# moz and uga number the columns differently
Q[2].column_dict = {'processor':1, 'seller':2,'services':6,'rent_out_land':7,
                    'other' : 8, 'no_other_way' : 9}
Q[5].column_dict = {'processor':1, 'seller':2,'services':6,'rent_out_land':7,
                    'other' : 8, 'no_other_way' : 9}


encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Do you receive income from any of the following?',
        label = 'H10',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'govt_benefits':1, 'remittances':2,'other_benefits':3,
                       'labor_for_hire':4,'sell_belongings' : 5, 'other' : 6},
        country = country,
        survey = 'rr'
        )
        for country in countries]

# moz and uga number the columns differently
Q[2].column_dict = {'govt_benefits':1, 'remittances':10,'other_benefits':13,
                    'labor_for_hire':14,'sell_belongings' : 15, 'other' : 17}
Q[5].column_dict = {'govt_benefits':1, 'remittances':10,'other_benefits':13,
                    'labor_for_hire':14,'sell_belongings' : 15, 'other' : 17}

encode_and_add(Q)

#_____________________________________________________________________________
Q = [Question_Encoder (
        qtype = 'multi',
        text =  'You said you receive a payment from the government (benefits, welfare, stipend, grant or another payment). How do you usually get this payment?',
        label = 'H11',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'direct_deposit':1, 'cash':2,'cheque':3,
                       'courier':4,'own_m_money' : 5, 'agent_m_money': 6,
                       'other_m_money': 7, 'digital_card':8, 'moneygram':9,                
                       'other' : 10},
        country = country,
        survey = 'rr'
        )
        for country in countries]


Q[0].column_dict = {'direct_deposit':1, 'cash':2,'cheque':3,'courier':4,'own_m_money' : 5, 
                    'digital_card':6, 'moneygram':7, 'other':8}
Q[2].column_dict = {'direct_deposit':1, 'cash':2,'cheque':3,'courier':4,
                    'own_m_money' : 6, 'agent_m_money': 7,'other_m_money': 8, 
                    'digital_card':9, 'moneygram':10,  'other' : 11}
Q[5].column_dict = {'direct_deposit':1, 'cash':2,'cheque':3,'courier':4,
                    'own_m_money' : 6, 'agent_m_money': 7,'other_m_money': 8, 
                    'digital_card':9, 'moneygram':10,  'other' : 11}

encode_and_add(Q)


#_____________________________________________________________________________
Q = [Question_Encoder (
        qtype = 'multi',
        text =  'How often do you make each of the following expenses?',
        label = 'H14',
        answers = {'weekly' : 1, 'monthly': 2, 'few_times_a_year':3,
                   'annually' : 4, 'according_to_harvest':5, 'never':6},
        column_dict = {'groceries':1, 'transportation':2,'medical': 3, 
                       'education':4, 'bills':5, 'emergencies':6, 
                       'investments': 7,'large_purchases': 8, 'home_repairs':9, 
                       'other' : 10},
        country = country,
        survey = 'rr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________
# I recoded the answers 4, 5 and 6 as np.nan

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Can you read any part of these sentences to me?',
        label = 'F62',
        answers = {'cannot_read' : 1, 'read_parts': 2, 'read_all':3},
        country = country,
        survey = 'rr'
        )
        for country in countries]

encode_and_add(Q)

##############################################################################
### Now we start the single-respondent survey

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Do you participate in household agricultural activities?',
        label = 'A99',
        answers = {'yes' : 1, 'no': 2},
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'How many years have you been farming?',
        label = 'A38',
        answers = {'less_than_two' : 1, '2_to_5': 2, '6_to_10': 3, 'more_than_10': 4},
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Do you intend to keep working in agriculture?',
        label = 'A39',
        answers = {'yes' : 1, 'no': 2},
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

# Mozambique allowed free-text answers to question A40, and there's no easy way 
# to summarize them, so I'll ignore A40 in subsequent analyses unless we really need it.

#_____________________________________________________________________________
Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Do you agree or disagree with the following statements',
        label = 'A41',
        answers = {'agree' : 1, 'disagree': 2},
        column_dict = {'enjoy_agriculture':1, 'want_ag_work_only':2,'want_expand': 3, 
                       'would_take_offered_job':4, 'am_satisfied':5, 'my_legacy':6, 
                       'make_ends_meet': 7,'want_children_continue': 8},
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________


# This is a BIG mess.  I need to check the xml metadata on the CGAP web site

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Are you a member of any of the following groups or associations?',
        label = 'A42',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'plant_weed_harvest':1, 'exporting':2, 'union': 3, 
                       'savings':4, 'women':5, 'processors':6, 
                       'cooperative': 7,'implements': 8},
        country = country,
        survey = 'sr'
        )
        for country in countries]

pre = {
    'plant_weed_harvest':1, 'exporting':2, 'union': 3, 'savings':4, 'women':5, 
    'processors':6, 'cooperative': 7,'implements': 8,}

Q[0].column_dict.update({'ICM_CLUB':9, 'water_users':10, 'livestock':11, 'other':12})
Q[1].column_dict.update({'water_users':9, 'livestock':10, 'other':11})

# The numeric codes in the moz User Guide might be wrong.  The column names don't skip.
# However, moz_sr A42_6, A42_7, A42_8 and A42_9 have 5 or fewer "yes" responses, so
# perhaps the User Guide numbers are telling us which column to select for each
# answer.  However, there is no 'A42_13'.  I will assume that moz_sr A42_1 to A42_8
# correspond with the answers in pre, above.  I can't tell which column corresponds 
# to 'caixa' or 'other', but I'll assume 'caixa' A42_11 and 'other' is A42_12. 
# But really, I don't know.  Perhaps I can download the xml metadata.   

# Also, tan has an 11th column but only 10 answers in the User Guide

Q[2].column_dict.update({'caixa':11, 'other':12})
Q[3].column_dict.update({'sacco':9, 'other':10})
Q[4].column_dict.update({'sacco':9, 'other':10})
Q[5].column_dict.update({'sacco':9, 'other':10})
    

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'What types of services do you get from these groups or associations?',
        label = 'A43',
        answers = {'agree' : 1, 'disagree': 2},
        column_dict = {'business_advice':1, 'farming_advice':2,'marketing': 3, 
                       'farm_implements':4, 'buy_on_credit':5, 'start_up':6,
                       'financial_advice':7, 'access_to_inputs': 8, 'storage': 9, 
                       'access_to_loans':10, 'profit_share':11, 'savings_account':12, 
                       'other': 13, 'none': 14},
        country = country,
        survey = 'sr'
        )
        for country in countries]

# Another case where the numeric codes for moz don't make sense, skipping 4, 9, 10,
# 14 and 15, even though in has columns named A43_1...A43_15, consecutively. I'll
# assume that the first 14 text answers correspond with the first 14 columns 

# Nigeria and Tanzania have an additional item
change_dict(Q[3].column_dict,['other','none'],[{'insurance':13,'other':14,'none':15}])
change_dict(Q[4].column_dict,['other','none'],[{'insurance':13,'other':14,'none':15}])

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  ' How often do you use each of the following sources of information for agricultural activities?',
        label = 'A44',
        answers = {'daily' : 1, 'weekly': 2, 'monthly' : 3 , 'less_than_monthly' : 4, 'never' : 5},
        column_dict = {'sms':1, 'radio':2,'television': 3, 
                       'internet':4, 'print_media':5, 'friends_family':6,
                       'religious_leader':7, 'community': 8, 'development': 9, 
                       'teachers':10, 'government':11, 'suppliers':12, 
                       'merchants': 13, 'extension': 14},
        country = country,
        survey = 'sr'
        )
        for country in countries]

Q[0].column_dict.update({'UIC':15,'middlemen':16,'other':17})
Q[1].column_dict.update({'middlemen':15,'cooperative':16,'other':17})
for i in [2,3,4,5]: Q[i].column_dict.update({'middlemen':15,'other':16})

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'How important is it to keep money aside for the following agricultural needs—very, somewhat, or not important?',
        label = 'A47',
        answers = {'very_important' : 1, 'somewhat_important': 2, 'not_important' : 3},
        column_dict = {'fertilizer':1, 'seeds':2,'pesticides': 3, 'equipment':4, 
                       'fuel':5, 'workers':6, 'security':7, 'investment': 8, 
                       'storage': 9, 'irrigation':10},
        country = country,
        survey = 'sr'
        )
        for country in countries]

for i in [0,1,3,4]: Q[i].column_dict.update({'transportation':11,'machinery':12,'other':13})
for i in [2,5]: Q[i].column_dict.update({'other':11})

encode_and_add(Q)


#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Do you currently keep money aside for the following agricultural needs?',
        label = 'A48',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'fertilizer':1, 'seeds':2,'pesticides': 3, 'equipment':4, 
                       'fuel':5, 'workers':6, 'security':7, 'investment': 8, 
                       'storage': 9, 'irrigation':10},
        country = country,
        survey = 'sr'
        )
        for country in countries]

for i in [0,1,3,4]: Q[i].column_dict.update({'transportation':11,'machinery':12,'other':13})
for i in [2,5]: Q[i].column_dict.update({'other':11})

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Do you want to keep money aside for the following agricultural needs?',
        label = 'A49',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'fertilizer':1, 'seeds':2,'pesticides': 3, 'equipment':4, 
                       'fuel':5, 'workers':6, 'security':7, 'investment': 8, 
                       'storage': 9, 'irrigation':10},
        country = country,
        survey = 'sr'
        )
        for country in countries]

for i in [0,1,3,4]: Q[i].column_dict.update({'transportation':11,'machinery':12,'other':13})
for i in [2,5]: Q[i].column_dict.update({'other':11})

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Do you currently store any of your crops after the harvest?',
        label = 'A52',
        answers = {'yes' : 1, 'no': 2},
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

# Nigeria treats A55 as a multi-answer question whereas the others treat it
# as a single-answer, free-text question. I won't encode it. 

# Q = [Question_Encoder (
#         qtype = 'single',
#         text =  'Where do you currently store your crops?',
#         label = 'A55',
#         country = country,
#         survey = 'sr'
#         )
#         for country in countries]

# encode_and_add(Q)
#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Why do you store your crops?',
        label = 'A56',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'better_price':1, 'minimize_risk':2,'sell_after_season': 3, 
                       'school_fees':4, 'major_expense':5, 'consume_later':6, 'other':7},
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Why do you not currently store your crops?',
        label = 'A57',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'no_storage':1, 'expensive':2,'no_leftover': 3, 
                       'bad_idea':4, 'need_money':5, 'other':6},
        country = country,
        survey = 'sr'
        )
        for country in countries]

# Once again, moz numeric codes do not correspond with its column names.
# cdi, moz, tan and uga have seven columns but only 6 question options. I will
# assume that 'A57_7' = 'A57_98'
encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Have you ever purchased livestock as an investment?',
        label = 'A58',
        answers = {'yes' : 1, 'no': 2},
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Do you currently have livestock that are investments?',
        label = 'A59',
        answers = {'yes' : 1, 'no': 2},
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Which of the following factors pose the most significant risk to your agricultural activities?',
        label = 'A60',
        country = country,
        survey = 'sr'
        )
        for country in countries]

for i in [0,1,3,4]:
    Q[i].answers = {
        'weather':1, 'power':2,'prices': 3, 'inputs':4, 'pests_disease':5, 
        'contract_broken':6, 'no_sale':7, 'perils_accidents': 8, 'health':9,
        'loss_of_land': 10, 'equipment_breakdown':11, 'input_quality':12, 
        'fuel_prices':13,'other':14}

# moz again!  I don't believe the following (i.e., them skipping code 3). Check the meta-data.
Q[2].answers = {
        'weather':1, 'power':2, 'prices': 4, 'inputs':5, 'pests_disease':6, 
        'contract_broken':7, 'no_sale':8, 'perils_accidents': 9, 'health':10,
        'loss_of_land': 11, 'other':12}

Q[5].answers = {
        'weather':1, 'power':2, 'prices': 3, 'inputs':4, 'pests_disease':5, 
        'contract_broken':6, 'no_sale':7, 'perils_accidents': 8, 'health':9,
        'loss_of_land': 10, 'equipment_breakdown':11, 'other':12}

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Have your agricultural activities been seriously affected by any of the following events in the past three years?',
        label = 'A61',
        answers = {'yes' : 1, 'no': 2},
        column_dict = {'weather':1, 'pests_disease':2,'accident': 3, 'market_prices':4, 
                       'input_prices':5, 'contract_broken':6, 'downturn_no_sale':7, 
                       'equipment_breakdown': 8, 'health':9},
        country = country,
        survey = 'sr'
        )
        for country in countries]

for i in [0,1,3,4]:
    Q[0].column_dict.update({'death':10,'unrest_or_war':11,'DK':12})

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'How did you mainly cope when this happened?',
        label = 'A62',
        answers = {'temp_job':1, 'took_loan':2, 'borrowed': 3, 'sold_livestock':4, 
                       'sold_asset':5, 'used_savings':6, 'insurance_paid':7, 
                       'no_need': 8, 'did_nothing':9},
        column_dict = {'weather':1, 'pests_disease':2,'accident': 3, 'market_prices':4, 
                       'input_prices':5, 'contract_broken':6, 'downturn_no_sale':7, 
                       'equipment_breakdown': 8, 'health':9},
        country = country,
        survey = 'sr'
        )
        for country in countries]

for i in [0,1,3,4]:
    Q[0].column_dict.update({'death':10,'unrest_or_war':11,'DK':12})

encode_and_add(Q)

#_____________________________________________________________________________
# This one has lots of inconsistencies.  Coding each column_dict separately.

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'When it comes to financial or income-related advice, who do you regularly talk to?',
        label = 'H16',
        column_dict = {},
        country = country,
        survey = 'sr'
        )
        for country in countries]

for i in [0,1]:
    Q[i].column_dict = {
        'chief':1, 'lead_farmer':2,'other_leader': 3, 'cooperative':4, 'extension':5, 
        'financial_inst':6, 'middlemen':7, 'savings_credit_inst': 8, 'other_community':9,
        'friends_family': 10, 'spouse':11, 'DK_who':12, 'have_noone' : 13, 'dont_seek_advice': 14}

for i in [3,4]:
    Q[i].column_dict = {
        'chief':1, 'local_councilor': 2, 'lead_farmer':3,'other_leader': 4, 'cooperative':5, 'extension':6, 
        'financial_inst':7, 'middlemen':8, 'savings_credit_inst':9, 'other_community':10,
        'friends_family': 11, 'spouse':12, 'DK_who':13, 'have_noone' : 14, 'dont_seek_advice': 15}
        
Q[2].column_dict = {
        'chief':1, 'lead_farmer':2,'other_leader': 3, 'cooperative':4, 'extension':5, 
        'financial_inst':6, 'middlemen':7, 'savings_credit_inst': 8, 'other_community':9,
        'friends_family': 10, 'DK_who':11, 'have_noone' : 12, 'dont_seek_advice': 13}    

Q[5].column_dict = {
        'chief':1, 'local_councilor':2, 'lead_farmer':3,'other_leader': 4, 'cooperative':5, 'extension':6, 
        'financial_inst':7, 'middlemen':8, 'savings_credit_inst': 9, 'other_community':10,
        'friends_family': 11, 'DK_who':12, 'have_noone' : 13, 'dont_seek_advice': 14}    
  
encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  ' In your opinion, how important is it for your household to save for each of the following?',
        label = 'H17',
        answers = {'very_important':1,'somewhat_important':2,'not_important':3},
        column_dict = {'future_purchases':1,'unexpected_event':2,'regular_purchases':3,
                       'school_fees':4,},
        country = country,
        survey = 'sr'
        )
        for country in countries]

Q[1].column_dict.update({'marriage':5,'funeral':6})
Q[3].column_dict.update({'marriage':5,'healthcare':6,'death':7,'loss_of_income':8})
Q[4].column_dict.update({'marriage':5,'healthcare':6,'death':7,'loss_of_income':8})
  
encode_and_add(Q)


#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  ' Which of the following do you feel your household needs to save for the most?',
        label = 'H18',
        answers = {'future_purchases':1,'unexpected_event':2,'regular_purchases':3,
                       'school_fees':4,},
        country = country,
        survey = 'sr'
        )
        for country in countries]

Q[0].answers.update({'other':5})
Q[1].answers.update({'marriage':5,'funeral':6})
Q[3].answers.update({'marriage':5,'healthcare':6,'death':7,'loss_of_income':8})
Q[4].answers.update({'marriage':5,'healthcare':6,'death':7,'loss_of_income':8})
  
encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  ' In your opinion, how important is it for your household to save at each of the following?',
        label = 'H19',
        answers = {'very_important':1,'somewhat_important':2,'not_important':3},
        column_dict = {'financial_inst':1,'informal_group':2,'at_home':3,'on_mobile':4},
        country = country,
        survey = 'sr'
        )
        for country in countries]

Q[1].column_dict.update({'with_collector':5})
  
encode_and_add(Q)

#_____________________________________________________________________________
# Uganda and Mozambique didn't ask question H20 and it is only a variant of H19,
# so I won't encode it. 

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'In your opinion, how important is it for your household to invest in each of the following?',
        label = 'H21',
        answers = {'very_important':1,'somewhat_important':2,'not_important':3},
        column_dict = {'farm':1,'home':2,'education':3},
        country = country,
        survey = 'sr'
        )
        for country in countries]

Q[0].column_dict.update({'business':4})
Q[1].column_dict.update({'business':4})
Q[3].column_dict.update({'business':4,'healthcare':5})
Q[4].column_dict.update({'business':4,'healthcare':5})

  
encode_and_add(Q)

#_____________________________________________________________________________
# H23, H24 and H26 are asked to assess whether farmers want banking services:
# H23: In the past 12 months, have you saved money with any of the following groups?
# H24: What would make you most likely to save money with a bank?
# H26_1 : When my money is in an account, it is constantly working for me
# H26_2 : I like to store money somewhere for a specific purpose
# ...

# I'm not encoding these

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Now I would like to ask you a few questions about how you manage your money.',
        label = 'H25',
        answers = {'yes':1,'no':2},
        column_dict = {'could_relatives_help':1,'household_skip_meal':2,'house_unlit':3,
                       'too_sick_to_work': 4, 'receive_support': 5},
        country = country,
        survey = 'sr'
        )
        for country in countries]


encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  ' Imagine that you have an emergency and you need to pay <<a lot>>. How possible is it that you could come up with <<a  lot>> within the next month—very possible, somewhat possible, or not possible?',
        label = 'H27',
        answers = {'very_possible':1,'somewhat_possible':2,'not_possible':3},
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'What would be the main source of money that you would use to come up with <<a lot>> within the next month?',
        label = 'H28',
        answers = {'savings':1,'friends_family':2,'working':3,'employer_loan':4,
                   'credit_card':5,'financial_inst':6, 'savings_credit_group':7,
                   'money_lender':8,'other':9},
        country = country,
        survey = 'sr'
        )
        for country in countries]

change_dict(Q[4].answers,['other'],[{'mobile_credit':9,'other':10}])
change_dict(Q[5].answers,['other'],[{'mobile_credit':9,'other':10}])

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'I will read several statements. Please, tell me how often does the following apply to you?',
        label = 'H30',
        answers = {'always_or_mostly':1,'sometimes':2,'rarely':3,'never':4},
        column_dict = {'income_exceeds_outgoing?':1,'fund_for_unplanned_expenses':2,
                       'pay_bills_ontime':3, 'savings_exceed_debts': 4},
        country = country,
        survey = 'sr'
        )
        for country in countries]


encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  ' Do you have any of the following?',
        label = 'H31',
        answers = {'yes':1,'no':2},
        column_dict = {'insurance':1,'living_will':2,'retirement':3, 
                       'savings': 4, 'investment': 5},
        country = country,
        survey = 'sr'
        )
        for country in countries]

Q[1].column_dict.update({'property_or_house':6})
Q[3].column_dict.update({'None':6})

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Which of the following types of insurance do you have?',
        label = 'H32',
        answers = {'yes':1,'no':2},
        column_dict = {'medical':1,'life':2,'car':3,'agriculture': 4, 'house': 5,
                       'unemployment':6, 'other':7 },
        country = country,
        survey = 'sr'
        )
        for country in countries]

for i in [0,1,2,5]: Q[i].column_dict.update({'None':8})

change_dict(Q[3].column_dict,['other'],[{'livestock':7,'funeral':8,'None':9}])
change_dict(Q[4].column_dict,['other'],[{'livestock':7,'funeral':8,'None':9}])

encode_and_add(Q)

#_____________________________________________________________________________

# H33 is coded as a single-answer question in all countries except moz. I can't
# make moz's multiple answers into a single answer, so I won't code the question.
# It isn't intersting, anyway: It just asks which type of insurance does the
# household need most.

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Does your family have a plan to manage the unexpected expenses, which might result from the following?',
        label = 'H34',
        answers = {'yes':1,'no':2},
        column_dict = {'loss_of_house':1,'major_medical':2,'bankruptcy_lost_job':3,
                       'loss_of_harvest': 4, 'loss_of_property': 5,
                       'death':6, 'time_without_food':7 },
        country = country,
        survey = 'sr'
        )
        for country in countries]

Q[3].column_dict.update({'crop_failure':8})
Q[4].column_dict.update({'crop_failure':8})

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'In the past 12 months, have you experienced any of the following events?',
        label = 'H35',
        answers = {'yes':1,'no':2},
        column_dict = {'medical':1,'death':2,'income_loss':3,'job_loss': 4, 
                       'wage_loss': 5, 'wedding':6, 'construction':7,'relocation':8 },
        country = country,
        survey = 'sr'
        )
        for country in countries]

Q[0].column_dict.update({'birth':9, 'none':10, 'DK':11})
Q[1].column_dict.update({'birth':9, 'none':10, 'DK':11})
Q[2].column_dict.update({'none':9, 'DK':10})
Q[3].column_dict.update({'birth':9, 'crop_failure':10, 'none':11})
Q[4].column_dict.update({'birth':9, 'crop_failure':10, 'none':11})
Q[5].column_dict.update({'none':9, 'DK':10})

encode_and_add(Q)


#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Do you agree or disagree with the following statements?',
        label = 'H37',
        answers = {'agree':1,'disagree':2},
        column_dict = {'actions_determine':1,'self_determine':2,'short_term':3,
                       'live_for_today': 4, 'future_determine_itself': 5, 
                       'work_hard':6, 'what_happens_happens':7,'power_determines':8 },
        country = country,
        survey = 'sr'
        )
        for country in countries]



encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Do you agree or disagree with the following statements?',
        label = 'H38',
        answers = {'agree':1,'disagree':2},
        column_dict = {'among_the_best':1,'think_hard_first':2,'unwise_to_plan':3,
                       'impulsive': 4, 'speak_before_think': 5, 
                       'opportunistic':6, 'have_aspirations':7,},
        country = country,
        survey = 'sr'
        )
        for country in countries]



encode_and_add(Q)



#_____________________________________________________________________________
# H42 isn't asked in moz or uga, so I'm not coding it here. 
#_____________________________________________________________________________

### Financial questions from the SR survey


Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Have you ever used any of the following financial services?',
        label = 'F46',
        answers = {'yes':1,'no':2},
        column_dict = {},
        country = country,
        survey = 'sr'
        )
        for country in countries]

Q[0].column_dict.update({'VLSA' : 1, 'other_informal': 2, 'money_guard' : 3,
                       'savings_collector' : 4, 'hawla_hundi' : 5, 'digital_card' : 6})
Q[1].column_dict.update({'VLSA' : 1, 'ROSCA' : 2, 'money_guard' : 3, 'savings_collector' : 4, 
                          'digital_card' : 5})
Q[2].column_dict.update({'VLSA' : 1, 'money_guard' : 2, 'savings_collector' : 3, 
                         'digital_card' : 4, 'money_lender':5})
Q[3].column_dict.update({'ROSCA' : 1, 'money_guard' : 2, 'savings_collector' : 3, 
                         'shopkeeper' : 4, 'digital_card' : 5})
Q[4].column_dict.update({'ROSCA' : 1, 'money_guard' : 2, 'savings_collector' : 3, 
                         'shopkeeper' : 4, 'digital_card' : 5})
Q[5].column_dict.update({'VLSA' : 1, 'ROSCA' : 2, 'chama' : 3, 'other_informal': 4, 'money_guard' : 5,
                       'savings_collector' : 6, 'shopkeeper' : 7, 'digital_card' : 8,
                       'money_lender' : 9})

encode_and_add(Q)


#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'multi',
        text =  'When did you last use any of the following financial services?',
        label = 'F47',
        answers = {'yesterday':1,'last_week':2, 'last_month': 3,
                   'last_quarter':4, 'before_last_quarter':5, 'stopped_using':6},
        column_dict = {},
        country = country,
        survey = 'sr'
        )
        for country in countries]

Q[0].column_dict.update({'VLSA' : 1, 'other_informal': 2, 'money_guard' : 3,
                       'savings_collector' : 4, 'hawla_hundi' : 5, 'digital_card' : 6})
Q[1].column_dict.update({'VLSA' : 1, 'ROSCA' : 2, 'money_guard' : 3, 'savings_collector' : 4, 
                          'digital_card' : 5})
Q[2].column_dict.update({'VLSA' : 1, 'money_guard' : 2, 'savings_collector' : 3, 
                         'digital_card' : 4, 'money_lender':5})
Q[3].column_dict.update({'ROSCA' : 1, 'money_guard' : 2, 'savings_collector' : 3, 
                         'shopkeeper' : 4, 'digital_card' : 5})
Q[4].column_dict.update({'ROSCA' : 1, 'money_guard' : 2, 'savings_collector' : 3, 
                         'shopkeeper' : 4, 'digital_card' : 5})
Q[5].column_dict.update({'VLSA' : 1, 'ROSCA' : 2, 'chama' : 3, 'other_informal': 4, 'money_guard' : 5,
                       'savings_collector' : 6, 'shopkeeper' : 7, 'digital_card' : 8,
                       'money_lender' : 9})

encode_and_add(Q)



# We need to recode the answers to single-answer question F50

data_dict['bgd_sr']['F50'].replace({1:1, 2:3, 3:4, 4:5, 5:3, 6:7},inplace=True)
data_dict['cdi_sr']['F50'].replace({1:1, 2:2, 3:4, 4:5, 5:7},inplace=True)
data_dict['moz_sr']['F50'].replace({'1':1, '2':4, '3':5, '5':7, '6':8, 'Other (specify)':np.nan}, inplace=True)
data_dict['nga_sr']['F50'].replace({1:2, 2:4, 3:5, 4:6, 5:7},inplace=True)
data_dict['tan_sr']['F50'].replace({1:2, 2:4, 3:5, 4:6, 5:7},inplace=True)
data_dict['uga_sr']['F50'].replace({1:1, 2:2, 3:4, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8},inplace=True)


Q = [Question_Encoder (
        qtype = 'single',
        text =  'Which of these service providers or services is the most important to you?',
        label = 'F50',
        answer_dict = {'VLSA' : 1, 'ROSCA' : 2, 'other_informal': 3, 'money_guard' : 4,
                       'savings_collector' : 5, 'shopkeeper' : 6, 'digital_card' : 7,
                       'money_lender':8},
        country = country,
        survey = 'sr'
        )
        for country in countries]


encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multiple',
        text =  'Which of the following services do these groups provide?',
        label = 'F49',
        column_dict = {'merry_go_round' : 1, 'lend_nonmembers' : 2, 'lend_members': 3, 'buy_for_members' : 4,
                       'guarantor_security' : 5, 'invest' : 6, 'purchase_tools' : 7,
                       'purchase_fixed_assets':8, 'funeral_emergency': 9, 'help_save':10},
        answer_dict = {'yes':1, 'no': 2},
        country = country,
        survey = 'sr'
        )
        for country in countries]




encode_and_add(Q)


#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multiple',
        text =  'Why do you not have a membership with any of these groups?',
        label = 'F51',
        column_dict = {'have_formal_account' : 1, 'have_no_money' : 2, 'stealing': 3, 
                       'unfamilar' : 4, 'no_need' : 5, 'no_trust' : 6, 'time_meeting' : 7},
        answer_dict = {'yes':1, 'no': 2},
        country = country,
        survey = 'sr'
        )
        for country in countries]




encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multiple',
        text =  'What would be the main reasons for borrowing money?',
        label = 'F53',
        column_dict = {'start_business' : 1, 'cash_flow' : 2, 'buy_inputs': 3, 
                       'big_purchases' : 4, 'other_agriculture' : 5, 'emergency' : 6, 
                       'school_fees' : 7,'daily_expenses' : 8},
        answer_dict = {'yes':1, 'no': 2},
        country = country,
        survey = 'sr'
        )
        for country in countries]




encode_and_add(Q)

#_____________________________________________________________________________

labels = ['F54','F55','F56']
answers = [{'very important':1, 'somewhat_important': 2, 'not_important': 3},
                          {'yes':1, 'no': 2},{'yes':1, 'no': 2}]
texts = ['For your agricultural activities, how important to you is it to borrow from the following?',
         'In the past 12 months have you attempted to borrow from the following?',
         'If the need arose, would you attempt to borrow from the following?']
         


for label,answers,text in zip(labels,answers,texts):
    

    Q = [Question_Encoder (
            qtype = 'multiple',
            text =  text,
            label = label,
            column_dict = None,
            answer_dict = answers,
            country = country,
            survey = 'sr'
            )
            for country in countries]
    
    Q[0].column_dict = {'bank': 1, 'microfinance':2, 'cooperative':3,'savings_collector':4,
                        'VLSA':5,'friends_family':6}
    Q[1].column_dict = {'bank': 1, 'microfinance':2, 'savings_collector':3,
                        'VLSA':4,'friends_family':5}
    Q[2].column_dict = {'bank': 1, 'microfinance':2, 'cooperative' : 3, 'SACCO':4,
                        'moneylender' : 5, 'friends_family' : 6}
    Q[3].column_dict = {'bank': 1, 'microfinance':2,  'SACCO':3, 'cooperative' : 4,
                        'moneylender' : 5, 'VSLA': 6, 'friends_family' : 7}
    Q[4].column_dict = {'bank': 1, 'microfinance':2,  'SACCO':3, 'cooperative' : 4,
                        'moneylender' : 5, 'VSLA': 6, 'friends_family' : 7}
    Q[5].column_dict = {'bank': 1, 'microfinance':2,  'SACCO':3, 'cooperative' : 4,
                        'moneylender' : 5,'friends_family' : 6}
    
    encode_and_add(Q)


#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Do you currently have any loans?',
        label = 'F58',
        answers = {'yes':1, 'no': 2},
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

""" The following items ask whether people think various financial products
are important, whether they have them, and whether they want them.  However,
very few people have any of them, and there is a bewildering number to choose 
from.  The main thing we want to know is whether people have loans and savings
plans etc.  So question F60 is the one to focus on.  I'll create a new column 
for each country that encodes the number of financial instruments that each
person has, then I'll make a data object for that."""

import copy

def count_instruments (country, columns, label):
    country_data = data_dict[country+'_sr']
    df = copy.copy(country_data[columns])
    df.replace({2:0},inplace=True) # 2 means they DON'T have the instrument
    s = df.sum(axis=1) # sum the 1 values, meaning they DO have it, across rows
    data_dict[country+'_sr'][label] = s

for c in ['bgd','nga','tan','uga']:
    count_instruments(c,['F60_1','F60_2','F60_3','F60_4','F60_5','F60_7'], 'FORMAL_LOANS')

count_instruments('cdi',['F60_1','F60_2','F60_3','F60_4','F60_5','F60_7','F60_16'], 'FORMAL_LOANS')
count_instruments('moz',['F60_1','F60_2','F60_4'], 'FORMAL_LOANS')

for c in ['bgd','cdi','nga','tan','uga']:
    count_instruments(c,['F60_8','F60_9'], 'SCHOOL_FEE_PLANS')
    count_instruments(c,['F60_10','F60_11'], 'AG_INPUTS_PLANS')

count_instruments('moz',['F60_5','F60_6'], 'SCHOOL_FEE_PLANS')
count_instruments('moz',['F60_7','F60_8'], 'AG_INPUTS_PLANS')

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'How many loans do you currently have?',
        label = 'FORMAL_LOANS',
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'How many credit/savings plans for school fees do you currently have?',
        label = 'SCHOOL_FEE_PLANS',
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'How many savings/payment plans for agricultural inputs do you currently have?',
        label = 'AG_INPUTS_PLANS',
        country = country,
        survey = 'sr'
        )
        for country in countries]

encode_and_add(Q)




##############################################################################
### Household data

Q = [Question_Encoder (
        qtype = 'single',
        text =  'The cluster, or smallest geographic locale, typically around 15 households',
        label = 'HH1',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'An intermediate size (between HH1 and HH6) geographic locale. Size ranges from 15 to 234 households.',
        label = 'HH7',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)
#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'The largest geographic locale. Size ranges from 80 to 800 households.',
        label = 'HH6',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'The country in which the survey was conducted',
        label = 'COUNTRY',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)
#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Whether the household is urban or rural',
        label = 'UR',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)
#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'A weight assigned to the household by the survey designers',
        label = 'HH_WEIGHT',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'The number of people in the household',
        label = 'HH10',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'The number of people in the household who are eligible to be interviewed',
        label = 'HH11',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'The language spoken at home',
        label = 'D14',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________
#See also H2, H2A and H2B etc and H8, above

Q = [Question_Encoder (
        qtype = 'single',
        text =  'What is your household’s smallest source of income?',
        label = 'D15',
        answers = {'regular_job':1, 'occasional_job':2,'retail_business':3,'services_business':4,
                       'grant_pension': 5, 'family_friends':6, 'growing_crops' : 7, 
                       'rearing_livestock':8, 'other' : 9},
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)


#_____________________________________________________________________________
#See also H2, H2A and H2B etc and H8, above

Q = [Question_Encoder (
        qtype = 'single',
        text =  'What is your household’s largest source of income?',
        label = 'D17',
        answers = {'regular_job':1, 'occasional_job':2,'retail_business':3,'services_business':4,
                       'grant_pension': 5, 'family_friends':6, 'growing_crops' : 7, 
                       'rearing_livestock':8, 'other' : 9},
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)


#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  'What is the minimum amount your household needs to survive per month (for personal expenses)?',
        label = 'D19',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  'Log transform of D19 (which is highly skewed)',
        label = 'D19_L',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  'D19_L is standardized (Z scores) within country Z scores can be compared across countries ',
        label = 'D19_LZ',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  'What is your household’s average monthly income across all sources of money that your household receives?',
        label = 'D21',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  'Log transform of D21 (which is highly skewed)',
        label = 'D21_L',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  'D21_L is standardized (Z scores) within country Z scores can be compared across countries ',
        label = 'D21_LZ',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  "Which answer best reflects your family's current financial situation?",
        label = 'D20',
        answers = {'not_enough_for_food':1, 'food_and_clothes_only' : 2,
                   'can_save_a_bit' : 3, 'can_buy_some_expensive_goods' : 4},
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Generally, who makes decisions on the following agricultural activities?',
        label = 'D22',
        answers = {'husband_boyfriend':1,'wife_girlfriend':2,'husband_and_wife':3,
                   'another_family_member':4,'NA': 5,'DK':6},
        column_dict = {'planting_time':1,'purchase_inputs':2,'harvest_time':3,
                       'amount_crops_to_sell': 4, 'when_where_to_sell': 5, 
                       'where_to_borrow':6, 'amount_livestock_to_sell':7,
                       'what_to_plant':8},
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)



#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'multi',
        text =  'Regardless of what you have, how important is it to your household to have the following?',
        note = "The answers have been re-ordered from not important to very important",
        label = 'D23',
        answers = {'not_important':1,'somewhat_important':2,'very_important':3},
        column_dict = {'bank_account':1,'mobile_phone':2,'mobile_money':3,
                       'insurance': 4, 'savings': 5, 'loan':6, 'credit':7},
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)
#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  'A derived, min-max scaled variable that represents the number of mobile phones in the household',
        label = 'MOBILES',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  'A binned version of MOBILES',
        label = 'MOBILES_B',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  'A derived, min-max scaled variable the quality of house construction materials',
        label = 'HOUSING0',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)


#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  'A derived, min-max scaled variable that includes water, toilet and energy in assessing the quality of the house',
        label = 'HOUSING1',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)


#_____________________________________________________________________________


Q = [Question_Encoder (
        qtype = 'single',
        text =  'A binned version of HOUSING1',
        label = 'HOUSING1_B',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'A derived, min-max scaled variable that combines questions about household possessions; an index of modest affluence.',
        label = 'POSSESS0',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'A version of POSSESS that includes bicycles, scooters and cars',
        label = 'POSSESS1',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'A binned version of POSSESS1',
        label = 'POSSESS1_B',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'The education level of the most educated person in the household',
        note = 'Education levels have already been rescaled to make them comparable across countries',
        label = 'D8_MAX',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'The average education level in the household',
        note = 'Education levels have already been rescaled to make them comparable across countries',
        label = 'D8_MEAN',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)


#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'The number of household members with age <= 15',
        label = 'NUM_KIDS',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)

#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'The number of household members with age > 15',
        label = 'NUM_ADULTS',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)


#_____________________________________________________________________________

Q = [Question_Encoder (
        qtype = 'single',
        text =  'Whether the household is above or below the poverty line',
        answers = {"above_poverty_line":0,"below_poverty_line":1},
        label = 'PPI_CUTOFF',
        country = country,
        survey = 'hh'
        )
        for country in countries]

encode_and_add(Q)



#_____________________________________________________________________________

filepath = '/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/CGAP/Data/Data Objects/'
cgap.write(filepath+'CGAP_JSON.txt')

