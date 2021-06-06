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
import random

COUNTRY='bgd'
GOLD="F58"
RANDOM_SEED=3252


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

all_countries = ['bgd','cdi','moz','nga','tan','uga']

sys.path.append('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/Code and Notebooks')
Data = CGAP_Decoded()
Data.read_and_decode('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/CGAP_JSON.txt')

x=Data.bgd_A1

bgd = Country_Decoded(COUNTRY,Data)

#get all data for the given country. then split it into train, dev, split
# qns_to_avoid=['COUNTRY','Country_Decoded']
# df1=bgd.concat_all_single_answer_qns(qns_to_avoid)
# df2=bgd.concat_all_multiple_answer_qns(qns_to_avoid)
# assert len(df1)==len(df2)
# df_combined = pd.concat([df1, df2], axis=1)
# df_combined=df_combined.fillna(-1)


qns_to_add=['F53','F54','F55','F56','F58']
df1=bgd.concat_all_single_answer_qns_to_add(qns_to_add)
df2=bgd.concat_all_multiple_answer_qns_to_add(qns_to_add)
df_combined = pd.concat([df1, df2], axis=1)
df_combined=df_combined.fillna(-1)


train,test_dev=train_test_split(df_combined,  test_size=0.2,shuffle=True)
test,dev=train_test_split(test_dev,  test_size=0.5,shuffle=True)



y_train_gold=np.asarray(train[GOLD]).reshape(-1, 1)
train.drop(GOLD,inplace=True,axis=1)
x_train_gold=np.asarray(train)

y_dev_gold=np.asarray(dev[GOLD])
dev.drop(GOLD,inplace=True,axis=1)
x_dev_gold=np.asarray(dev)

#MLP
model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#model=neighbors.KNeighborsClassifier()
#model = LogisticRegression()
#model = tree.DecisionTreeClassifier()
#model = RandomForestClassifier(n_estimators=10)
# #model = Perceptron(tol=1e-3, random_state=0)
#model = svm.SVC()



# Train the model using the training sets
model.fit(x_train_gold, y_train_gold)
y_dev_pred = model.predict(x_dev_gold)

print(classification_report(y_dev_gold, y_dev_pred))

#
# # Plot outputs
# plt.scatter(x_dev_gold, y_dev_gold, color='black')
# plt.scatter(x_dev_gold, y_dev_pred, color='blue', linewidth=3)
# plt.xlabel("farmers")
# plt.ylabel("Do you currently have any loans.1 yes 2 no")
# plt.xticks(())
# plt.yticks(())
# plt.show()