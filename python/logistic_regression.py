#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression
# ##  Spam Filtering with Sklearn

# In[33]:


import numpy as np
import pandas as pd
df = pd.read_csv('data/SMSSpamCollection/SMSSpamCollection', delimiter='\t', header=None)
print(df.head())
print ('Number of spam messages:', df[df[0] == 'spam'][0].count())
print ('Number of ham messages:', df[df[0] == 'ham'][0].count())


# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn import metrics 
from sklearn.metrics import classification_report


# In[35]:


df = pd.read_csv('data/SMSSpamCollection/SMSSpamCollection', delimiter='\t', header=None)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])


# In[36]:


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)


# In[37]:


classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
for prediction in predictions[:5]:
    print ('Prediction: %s.' % prediction)
    
print (X_test_raw[:5])


# ### Accuracy Metrics
# - Notes are somehere else, I'll just skip it here.

# In[38]:


from sklearn.metrics import accuracy_score
y_pred, y_true = [0, 1, 1, 0], [1, 1, 1, 1]
print ('Accuracy:', accuracy_score(y_true, y_pred))


# ### Precision and Recall
# - Precision and recall are two measurements usually used for evaluating classifiers due to their sensitivities of false/true negative/positive results.
# - Notes are in [Chapter5.html](Chap5-Basic_Practice.html)
# 
# ### The F1 Measure
# - the harmonic mean, or weighted average, of the precision and recall scores. 
# - It is also called the **f-measure** or the **f-score**.
# - It is calculated using the formula **F1= 2* P*R/(P+R)**
# - Penalizes classifiers with imbalanced precision and recall scores, like a classifier always returning positive predictions
# - If a model is perfectly balanced, the f1-score will be 1
# 
# ### AUC, Hyperparameter Tuning...
# - [Chapter5.html](Chap5-Basic_Practice.html)
# 

# In[ ]:




