#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
diabetes = datasets.load_diabetes()
diabetes["feature_names"]


# In[7]:


X=diabetes.data[:, np.newaxis, 2]
y=diabetes["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#print(X_train.shape,y_train.shape)

N_train=y_train.shape[0]
N_test=y_test.shape[0]


# In[8]:


def gradient_descent(m,b,learning_rate):
    for _ in range (0,N_train):
        x=X_train[_]
        y=y_train[_]
        
        y_hat=m*x+b
        error=y-y_hat
        
        m+=error*x*learning_rate
        b+=error*learning_rate
        
    return m,b
        
    


# In[10]:


def plot_training_data():
    x = np.linspace(-0.1, 0.18, 500)
    plt.plot(X_train, y_train, 'ro')
    plt.plot(x,m*x+b);
    plt.show()


# In[11]:


def plot_test_data():
    x = np.linspace(-0.1, 0.18, 500)
    plt.plot(X_test, y_test, 'ro')
    plt.plot(x,m*x+b);
    plt.show()


# In[12]:


m=1
b=0
iterations=100
for _ in range (0,iterations,1):
    m,b=gradient_descent(m,b,0.01)
    plot_test_data()

