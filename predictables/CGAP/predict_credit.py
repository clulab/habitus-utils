import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn import  linear_model
import matplotlib.pyplot as plt
from CGAP_JSON_Encoders_Decoders import CGAP_Decoded
from sklearn.neural_network import MLPClassifier
sys.path.append('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/Code and Notebooks')
Data = CGAP_Decoded()
Data.read_and_decode('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/CGAP_JSON.txt')
countries = ['bgd','cdi','moz','nga','tan','uga']
#all_rows=Data.col('uga','F58')
all_rows=Data.col('uga','A32')

assert all_rows is not None
assert len(all_rows)>0

all_rows=all_rows.dropna()
train,test_dev=train_test_split(all_rows,  test_size=0.2, shuffle=True)
test,dev=train_test_split(test_dev,  test_size=0.5, shuffle=True)

x_train=np.asarray(list(train.index))
y_train=np.asarray(list(train))
x_dev=np.asarray(list(dev.index))
y_dev=np.asarray(list(dev))

# Create linear regression object
model = linear_model.LinearRegression()


#MLP
#model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
# Train the model using the training sets
model.fit(x_train.reshape(-1, 1),y_train)
dev_y_pred = model.predict(x_dev.reshape(-1, 1))

# Plot outputs
plt.scatter(x_dev, y_dev,  color='black')
plt.scatter(x_dev, dev_y_pred,color='blue')

for index,(x,y) in enumerate(zip(x_dev, dev_y_pred)):
    if(index%5)==0:
        plt.annotate('(%s, %s)' % (x,y), xy=(x,y),xytext=(x,y+2), textcoords='data')


plt.xlabel("farmers")
plt.ylabel("Do you currently have any loans.1 yes 2 no")
plt.xticks(())
plt.yticks(())
plt.show()