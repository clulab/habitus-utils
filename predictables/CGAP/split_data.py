import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn import  linear_model,svm,neighbors, tree
import matplotlib.pyplot as plt
from CGAP_JSON_Encoders_Decoders import *
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
from utils import *
from sklearn.ensemble import RandomForestClassifier



sys.path.append('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/Code and Notebooks')
Data = CGAP_Decoded()
Data.read_and_decode('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/CGAP_JSON.txt')

countries = ['bgd','cdi','moz','nga','tan','uga']

#get all data for the given country. then split it into train, dev, split
qns_to_avoid=['COUNTRY','Country_Decoded']
df1=Data.concat_all_single_answer_qns(qns_to_avoid)
df2=Data.concat_all_multiple_answer_qns(qns_to_avoid)
assert len(df1)==len(df2)
df_combined = pd.concat([df1, df2], axis=1)
df_combined=df_combined.fillna(-1)


train,test_dev=train_test_split(df_combined,  test_size=0.2)
test,dev=train_test_split(test_dev,  test_size=0.5)



