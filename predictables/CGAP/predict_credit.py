import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn import  linear_model
import matplotlib.pyplot as plt
from CGAP_JSON_Encoders_Decoders import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pandas as pd


COUNTRY='bgd'
sys.path.append('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/Code and Notebooks')
Data = CGAP_Decoded()
Data.read_and_decode('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/CGAP_JSON.txt')
countries = ['bgd','cdi','moz','nga','tan','uga']
#do you have loan-use classifier
#all_rows=Data.col('uga','F58')
bgd = Country_Decoded(COUNTRY,Data)


#qtype=multi
# df = pd.concat([
#     Data.col('moz','A5','Rice'),
#     Data.col('moz','H28'), # "col" version
#     Data.moz_H28.df.H28,  # non="col" version: you have to say H28 twice
#         ], axis=1)


#for the list of answers available for tthis country
qns_to_avoid=['D19','A26','COUNTRY','Country_Decoded']
df1=bgd.concat_all_single_answer_qns(qns_to_avoid)
df2=bgd.concat_all_multiple_answer_qns(qns_to_avoid)
df_combined = pd.concat([df1, df2], axis=1)






#all_rows=df_combined.dropna()
train,test_dev=train_test_split(df_combined,  test_size=0.2, shuffle=True)
test,dev=train_test_split(test_dev,  test_size=0.5, shuffle=True)

x_train_gold=np.asarray(list(train.index))
y_train_gold=np.asarray(list(train))
x_dev_gold=np.asarray(list(dev.index))
y_dev_gold=np.asarray(list(dev))

# Create linear regression object
#model = linear_model.LinearRegression()


#MLP
model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

# Train the model using the training sets
model.fit(x_train_gold.reshape(-1, 1), y_train_gold)
y_dev_pred = model.predict(x_dev_gold.reshape(-1, 1))


print(classification_report(y_dev_gold, y_dev_pred))


# Plot outputs
plt.scatter(x_dev_gold, y_dev_gold, color='black')
plt.scatter(x_dev_gold, y_dev_pred, color='blue', linewidth=3)



plt.xlabel("farmers")
plt.ylabel("Do you currently have any loans.1 yes 2 no")
plt.xticks(())
plt.yticks(())
plt.show()