#!/usr/bin/env python
# coding: utf-8

# ## Multiple Regression

# In[154]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
y = [[7], [9], [13], [17.5], [18]]


# ### Write simple linear regression in the form of the matrix
# To include multiple explanatory variables, we write them in a matrix.!

# In[162]:


from numpy.linalg import inv
from numpy import dot
from numpy import transpose as trans

dot(inv(dot(trans(X),X)),dot(trans(X),y))


# ### Caulculate LSE using numpy

# In[163]:


# ## normalize
# from sklearn import preprocessing
# scaler=preprocessing.StandardScaler()
# scaler.fit(X_train)
# X_train=scaler.transform(X_train)

# y_max=np.max(y)
# y_min=np.min(y)
# y_range=y_max-y_min
# y_train=y_train/y_range


# In[165]:


from numpy.linalg import lstsq
print( lstsq(X,y)[0] )


# ### Predict with sk linear regression model

# In[171]:


lr_model=LinearRegression()
lr_model.fit(X,y)
print(lr_model.score(X,y))


# In[175]:


X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7],[9],[13],[17.5],[18]]
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11],   [8.5],  [15],    [18],    [11]]
lr_model.fit(X,y)
predictions=lr_model.predict(X)
for i, prediction in enumerate(predictions):
    print ('Predicted: %s, Target: %s' % (prediction, y_test[i]))
print ('R-squared: %.2f' % lr_model.score(X_test, y_test))

