#!/usr/bin/env python
# coding: utf-8

# VARIABLE IDENTIFICATION
#     INDEPENDENT VARIABLES:
#         1.OPEN
#         2.HIGH
#         3.LOW
#         4.VOLUME
#     DEPENDENT VARIABLES:
#         5.ADJ CLOSE
#         6.CLOSE
DATA TYPES
    integer:
        volume
        float:
        close
        high
        open
        low
        adjclose
        
# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[3]:


dp = pd.read_csv("AAPL1.csv")


# In[4]:


dp.head()


# In[5]:


dp.tail()


# In[6]:


dp.shape


# In[12]:


dp.index


# In[7]:


dp.describe()


# In[9]:


dp.columns


# In[10]:


dp.info() #iexist or nott will shows the datatypes present in the dataset and also shows null values


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


import matplotlib.dates as mdates


# In[11]:


plt.plot(dp.index,dp['Adj Close'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.grid(True)
plt.xticks(rotation=90)
plt.show()


# In[11]:


import seaborn as sns


# In[14]:


sns.boxplot(y='Adj Close',data=dp)#it is used to see the outliers


# In[15]:


sns.distplot(dp["Adj Close"],bins=10)


# In[16]:


sns.boxplot(y='Close',data=dp)


# In[17]:


sns.distplot(dp["Close"],bins=10)


# In[18]:


sns.boxplot(y='Open',data=dp)


# In[19]:


sns.distplot(dp["Open"],bins=10)


# In[20]:


sns.boxplot(y='High',data=dp)


# In[21]:


sns.distplot(dp["High"],bins=10)


# In[22]:


sns.boxplot(y='Low',data=dp)


# In[23]:


sns.distplot(dp["Low"],bins=10)


# In[24]:


sns.boxplot(y='Volume',data=dp)


# In[25]:


sns.distplot(dp["Volume"],bins=10)


# In[26]:


dp_con=dp.iloc[:,:-1]
sns.pairplot(dp_con)


# In[27]:


dp_con1=dp.iloc[::-1]
sns.pairplot(dp_con1)


# In[28]:


sns.heatmap(dp.corr(),annot=True,linewidth=0.5)


# In[29]:


plt.figure(figsize=(16,8))
plt.title('AAPL')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(dp['Close'])
plt.show()


# In[13]:


dp.isnull()


# In[14]:


dp.isnull().values.any()


# In[15]:


dp.shape


# In[15]:


dp.info()


# In[17]:


dp.isnull().sum()


# In[32]:


dp = dp['Adj Close']
dp


# In[18]:


#create a variable to predict 'x' days out into the future
future_days = 50
#create a new column (target) shifted 'x' units/days up
dp['Prediction'] = dp['Adj Close'].shift(-future_days)
dp.head(4)


# In[22]:


X=np.array(dp.drop(['Prediction'],1))[:-future_days]
print(X)


# In[23]:


#create the target dataset(y) and convert it to a numpy array and get all of the target values except the last 'x' rows/days
y=np.array(dp['Prediction'])[:-future_days]
print(y)


# In[24]:


#split the data 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[25]:


x_train.shape


# In[26]:


x_test.shape


# In[27]:


y_train.shape


# In[25]:


y_test.shape


# In[ ]:




