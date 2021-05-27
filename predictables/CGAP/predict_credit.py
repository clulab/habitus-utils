import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn import  linear_model
import matplotlib.pyplot as plt

sys.path.append('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/Code and Notebooks')

from CGAP_JSON_Encoders_Decoders import CGAP_Decoded


# Change this filepath to one for your machine. The actual file is on our Box
# folder at https://pitt.app.box.com/folder/136317983622

Data = CGAP_Decoded()
Data.read_and_decode('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/CGAP_JSON.txt')

countries = ['bgd','cdi','moz','nga','tan','uga']
print("done")


print(Data.col('uga','F58'))


all_rows=Data.col('uga','F58')
counter=0



train,test_dev=train_test_split(all_rows,  test_size=0.2, shuffle=True)
test,dev=train_test_split(test_dev,  test_size=0.5, shuffle=True)

print(f"train data shape :{type(train.shape)}")
print(f"train data index:{type(train.index)}")
x_train=np.asarray(list(train.index))
y_train=np.asarray(list(train))
x_dev=np.asarray(list(dev.index))
y_dev=np.asarray(list(dev))



# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train.reshape(-1, 1),y_train)
dev_y_pred = regr.predict(x_dev.reshape(-1, 1))



# Plot outputs
plt.scatter(x_dev, y_dev,  color='black')
plt.plot(x_dev, dev_y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()