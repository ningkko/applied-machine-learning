#!/usr/bin/env python
# coding: utf-8

# ## Linear classification

# In[222]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
import numpy as np

iris=datasets.load_iris()
# get only the first 2 attributes
X,y=iris.data[:,:2],iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

print (X_train[0],y_train[0],"\n",X_train.shape, y_train.shape)

iris.feature_names


# In[252]:


#X_train


# ## Standardization
# #### Standardize the features. Does not change the distribution, bt avoid features with extreme values, which may weight too much on the final results

# In[224]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_train_standardized=scaler.transform(X_train)
X_test_standardized=scaler.transform(X_test)

for _ in range (10):
    print(X_train[_],X_train_standardized[_])


# ## Visualize the data

# In[225]:


import matplotlib.pyplot as plt

def plot_iris_w_two_features(X_train_standardized, x_label,y_label):
    colors=['red','yellow','blue']
    for species in range(len(colors)):
        sepal_length=X_train_standardized[:,0][y_train==species]
        sepal_width=X_train_standardized[:,1][y_train==species]
        plt.scatter(sepal_length,sepal_width,c=colors[species])

    plt.legend(iris.target_names)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


# In[226]:


plot_iris_w_two_features(X_train_standardized, 'Sepal length [standardized]','Sepal width [standardized]')


# ## Stochastic gradient descent 
# #### To find a boundary line/ hyperplane by minimizing the loss function. Evey feature classification decision depends on one hyperplane.

# In[227]:


from sklearn.linear_model import SGDClassifier
clf=SGDClassifier()
clf.fit(X_train_standardized,y_train)


# #### If print out the coefficient, it will have 3 values rather than one. Because the classifier converts the 3-class binary problem into 3 binary problems - separate each class from the rest 2 classes. 
# #### It should also have 3 intercept values for the 3 solutions.

# In[228]:


print(clf.coef_)
print(clf.intercept_)


# ## Visualize result hyperplanes

# In[229]:


def visualize(X_train):
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() +.5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() +.5
    xs = np.arange(x_min, x_max, 0.5)
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(10, 6)
    for i in [0, 1, 2]:
        axes[i].set_aspect('equal')
        axes[i].set_title('Class '+ str(i) + ' versus the rest')
        axes[i].set_xlabel('Sepal length')
        axes[i].set_ylabel('Sepal width')
        axes[i].set_xlim(x_min, x_max) 
        axes[i].set_ylim(y_min, y_max)
        plt.sca(axes[i])
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,cmap=plt.cm.prism)
        ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
        plt.plot(xs, ys)


# In[230]:


visualize(X_train_standardized)


# ## Predict using this classifier
# 

# In[231]:


for species in range(len(colors)):
    sepal_length=X_train[:,0][y_train==species]
    sepal_width=X_train[:,1][y_train==species]
    plt.scatter(sepal_length,sepal_width,c=colors[species])

plt.legend(iris.target_names)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Sepal width [cm]')


# In[232]:


def predict_point(point):
    predicted_class=clf.predict(scaler.transform(point))
    color=colors[predicted_class[0]]
    print(color)
    
# point(8,4) should be blue
this_should_be_blue = [[8,4]]
# point(5,4.5) should be red
this_should_be_red = [[5,4.5]]

predict_point(this_should_be_blue)
predict_point(this_should_be_red)


# ## Evaluate the classifier

# In[233]:


from sklearn import metrics
y_train_prediction = clf.predict(X_train_standardized)
print(y_train_prediction)
print(y_train)
metrics.accuracy_score(y_train,y_train_prediction)


# #### It's easy to see that the classifier has some difficulty in differentiating category 1 and 2, as what we can observe from the graphs.

# ## Now test on the test dataset:

# In[234]:


y_test_prediction = clf.predict(X_test_standardized)
print(y_test_prediction)
print(y_test)
metrics.accuracy_score(y_test,y_test_prediction)


# #### A pretty bad job at predicting. And the main prediction mistake happens between prediction class 1 and 2. 

# ### Ways of evaluation
# #### Precision: #correct(positive) / #total positive
# #### Recall: #correct(positive) / #(correct(positive) + false(negative))
# The precision is the proportion of relevant documents in the list of all returned documents. The recall is the ratio of the relevant documents returned by the search engine to the total number of the relevant documents that could have been returned.
# 
# Precision can be used when solving a spam problem. People have more tolerance in not spamming junk mails than in spamming important emails. 
# #### F1-score:  2\*precision\*recall/(precision+recall)
# Tries to combine precision and recall

# In[235]:


print(metrics.classification_report(y_test,y_test_prediction,target_names=iris.target_names))


# ### The confusion metrics
# Useful when accessing multi-class classification.
# In its (i,j) cell, it shows the number of instances i that were predicted to be in class j.

# In[241]:


confusion_metrics = metrics.confusion_matrix(y_test,y_test_prediction)
print(confusion_metrics)


# ## Cross-validation
# #### Divides the dataset into several ones and build a model on each of them to avoid 'bias.'

# In[269]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

clf = Pipeline([
    ('scaler',StandardScaler()),
    ('linear_model',SGDClassifier())
])

# use 5 folds
cross_validation = KFold( 5, shuffle=True, random_state=33)
scores = cross_val_score(clf, X, y, cv=cross_validation)
print(scores)


# #### Calculate the mean and SE

# In[275]:


from scipy.stats import sem
print("Mean: ",np.mean(scores),"\nStandard Error: ",sem(scores))


# 
# 
# 
# # ==============================================================
# 
# 
# 

# ### If we have chosen the petal length and the petal width to be the feature

# In[242]:


X_new,y_new=iris.data[:,2:4],iris.target
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.25, random_state=33)

scaler_new = preprocessing.StandardScaler().fit(X_train_new)
X_train_standardized_new=scaler_new.transform(X_train_new)
X_test_standardized_new=scaler_new.transform(X_test_new)

plot_iris_w_two_features(X_train_standardized_new, 'Petal length [standardized]','Petal width [standardized]')
clf.fit(X_train_standardized_new,y_train_new)
visualize(X_train_standardized_new)

y_test_prediction_new= clf.predict(X_test_standardized_new)
print(y_test_prediction_new)
print(y_test_new)
print("accuracy: ",metrics.accuracy_score(y_test_new,y_test_prediction_new))
print(metrics.classification_report(y_test_new,y_test_prediction_new,target_names=iris.target_names))

confusion_metrics = metrics.confusion_matrix(y_test_new,y_test_prediction_new)
print(confusion_metrics)


# #### The classifier cannot find a way to separate class 1 with other 2 classes using only one line. But it knows clearer than when using sepal lengths as features about how to differentiate class 0 and class 2 from other classes. So overall this new classifier does a better job than the previous one. For the confusion metrics, we can see that only 3 out of total were predicted wrong.
# 

# In[278]:


clf_new = Pipeline([
    ('scaler',StandardScaler()),
    ('linear_model',SGDClassifier())
])

# use 5 folds
cross_validation = KFold( 5, shuffle=True, random_state=33)
scores = cross_val_score(clf, iris.data[:,2:4], y, cv=cross_validation)
print(scores)


# In[279]:


print(np.mean(scores))
print(sem(scores))


# In[ ]:




