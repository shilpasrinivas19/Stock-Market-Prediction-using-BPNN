#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv('AAPL1.csv')
dataset = ds.iloc[:, [2,5]].values

X = ds.iloc[:, 2].values
y = ds.iloc[:, 5].values


# In[3]:


df1=ds.reset_index()['Close']


# In[4]:


df1.shape


# In[5]:


plt.plot(df1)


# In[5]:


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

scaler  = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

X = dataset_scaled[:, 0]
y = dataset_scaled[:, 1]


# In[6]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[7]:


# Sizes of train_ds, test_ds
dataset_sz = X.shape[0]
train_sz = X_train.shape[0]
test_sz = X_test.shape[0]


# In[8]:


############ Building the ANN ############
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[9]:


# Initialising the ANN
regressor = Sequential()


# In[10]:


# Adding the input layer and the first hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1))
regressor.add(Dropout(.2))+


# In[11]:


# Adding the first hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(.2))


# In[12]:


# Adding the second hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(.2))


# In[13]:


# Adding the third hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(.2))


# In[14]:


# Adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[15]:


# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[16]:


# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 10, epochs = 100)


# In[17]:


############ Save & load Trained Model ############
# Save & load Trained Model
from keras.models import load_model


# In[18]:


# Save Trained Model
regressor.save('AAPL1.h5')


# In[19]:


# deletes the existing model
del regressor


# In[20]:


# load Trained Model
regressor = load_model('AAPL1.h5')


# In[21]:


############ Predict & Test the Model ############
real_stock_price = np.array(X_test)
inputs = real_stock_price
predicted_stock_price = regressor.predict(inputs)

# rebuild the Structure
dataset_test_total = pd.DataFrame()
dataset_test_total['real'] = real_stock_price
dataset_test_total['predicted'] = predicted_stock_price


# In[22]:


# real data price VS. predicted price
predicted_stock_price = scaler.inverse_transform(dataset_test_total) 

# Calc difference between real data price and predicted price
toler_rate = np.zeros(test_sz)

for i in range(0, test_sz):
    toler_rate[i] = abs(predicted_stock_price[i, 0] - predicted_stock_price[i, 1])

# tolerance threshold
toler_treshold = 5.0

# Wrong predicted count
err_cnt = 0
for i in range(0, test_sz):
    if abs(predicted_stock_price[i, 0] - predicted_stock_price[i, 1]) <= toler_treshold/100 * predicted_stock_price[i, 0] :
        pass
    else:
        err_cnt +=1

import math

# Calc MSE
mse = 0.0
for i in range(0, test_sz):
    mse += (predicted_stock_price[i, 0] - predicted_stock_price[i, 1])**2

mse /= test_sz


# In[23]:


############ Visualizing the results ############
all_real_stock_price = np.array(y)
inputs = np.array(X)
#inputs = np.reshape(inputs, (dataset_sz, 1, 1))
all_predicted_stock_price = regressor.predict(inputs)

# rebuild the Structure
dataset_test_total = pd.DataFrame()
dataset_test_total['real'] = all_real_stock_price
dataset_test_total['predicted'] = all_predicted_stock_price

# real test data price VS. predicted price
stock_price_predicted_real = scaler.inverse_transform(dataset_test_total) 

# Visualising the results
plt.plot(stock_price_predicted_real[:, 0], color = 'red', label = 'Real Stock Price')
plt.plot(stock_price_predicted_real[:, 1], color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price($)')
plt.savefig('stock_price_predicted_real.png')
plt.legend(['real', 'predicted'], loc='upper left')
plt.show()


plt.plot(stock_price_predicted_real[:, 0], color = 'red', label = 'Real Stock Price')
plt.title('Stock Prices')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price($)')
plt.savefig('Real_stock_price.png')
plt.show()


plt.plot(stock_price_predicted_real[:, 1], color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price($)')
plt.savefig('Predicted_stock_price.png')
plt.show()


# ####Evaluating the performance of Given Model by using MAE, MSE, R^2 Metrics####
# ->The smaller MSE or MAE value shows better results than larger Values
# ->R^2 metrics shows 0 and 1 for N0 FIT and PERFECT FIT respectively

# In[24]:


# Cross Validation Regression MAE
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
X = dataset_scaled[:,0:1]
Y = dataset_scaled[:,1]
kfold = model_selection.KFold(n_splits=2, random_state=1, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f" % (results.mean()))


# In[25]:


#Cross Validation Regression MSE
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f" % (results.mean()))


# In[26]:


#Cross Validation Regression R^2
scoring = 'r2'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f" % (results.mean()))


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# fit final model
regressor = LinearRegression()
regressor.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_regression(n_samples=30, n_features=2, noise=0.1, random_state=1)
# make a prediction
ynew = regressor.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))


# In[ ]:




