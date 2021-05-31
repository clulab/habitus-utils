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

COUNTRY='bgd'
#pick one of the columns as gold label- we are going to make machine predicttt thatt
#f58=does the farmer currenttly have any loans
GOLD="F58"



sys.path.append('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/Code and Notebooks')
Data = CGAP_Decoded()
Data.read_and_decode('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/CGAP_JSON.txt')

x=Data.bgd_A1
countries = ['bgd','cdi','moz','nga','tan','uga']
bgd = Country_Decoded(COUNTRY,Data)

#some qns are dependant on previous answers. or are just bookkeeping.-avoid them in training
#todo: do something about qns dependantt on previous answers. eg. A26
qns_to_avoid=['COUNTRY','Country_Decoded']
df1=bgd.concat_all_single_answer_qns(qns_to_avoid)
df2=bgd.concat_all_multiple_answer_qns(qns_to_avoid)
assert len(df1)==len(df2)
df_combined = pd.concat([df1, df2], axis=1)
df_combined=df_combined.fillna(-1)

train,test_dev=train_test_split(df_combined,  test_size=0.2)
test,dev=train_test_split(test_dev,  test_size=0.5)

y_train_gold=np.asarray(train[GOLD]).reshape(-1, 1)
train.drop(GOLD,inplace=True,axis=1)
x_train_gold=np.asarray(train)

y_dev_gold=np.asarray(dev[GOLD])
dev.drop(GOLD,inplace=True,axis=1)
x_dev_gold=np.asarray(dev)

#MLP
#model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#model = Perceptron(tol=1e-3, random_state=0)
#model = LogisticRegression()
#model = svm.SVC()
#model=neighbors.KNeighborsClassifier()
model = tree.DecisionTreeClassifier()
#model = RandomForestClassifier(n_estimators=10)


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