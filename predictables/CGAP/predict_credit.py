import sys
import os, json
import copy
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

sys.path.append('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/Code and Notebooks')

from CGAP_JSON_Encoders_Decoders import Question_Decoder, CGAP_Encoded, CGAP_Decoded, Country_Decoded


# Change this filepath to one for your machine. The actual file is on our Box
# folder at https://pitt.app.box.com/folder/136317983622

Data = CGAP_Decoded()
Data.read_and_decode('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/CGAP_JSON.txt')

countries = ['bgd','cdi','moz','nga','tan','uga']
print("done")



all_rows=Data.col('uga','F58')
counter=0

print(len(all_rows))

train,test=train_test_split(all_rows,  test_size=0.2, shuffle=True)
print(len(train))
print(len(test))