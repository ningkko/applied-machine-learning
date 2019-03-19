#!/usr/bin/env python
# coding: utf-8

# ### One-Hot Encoding
# - Contains order

# In[1]:


from sklearn.feature_extraction import DictVectorizer


# In[2]:


onehot_encoder = DictVectorizer()
instances = [
    {'city': 'New York'},
    {'city': 'San Francisco'},
    {'city': 'Chapel hILL'}
]
print(onehot_encoder.fit_transform(instances).toarray())


# ## Extracting features from text
# ### The bag-of-words representation
# - does not encode syntax, ignores the order of words, disregards all grammar
# - Effective for document classification and retrieval
# - Uses a feature vector with an element for each of the words
# - Tokens: case-insensitive words or affixes

# In[24]:


from sklearn.feature_extraction.text import CountVectorizer
corpus = [
        'UNC played Duke in basketball',
        'Duke lost the basketball game',
        'I ate a sandwich'
]
vectorizer = CountVectorizer()

## Prints whether the nth word in the vocabulary list has appeared
## in the nth feature vector
print(vectorizer.fit_transform(corpus).todense())
## Prints the vocabulary list
print(vectorizer.vocabulary_)


# #### We can calculate the Euclidean distance between each pair of the vectors, to see which ones are more similar. 

# In[25]:


from sklearn.metrics.pairwise import euclidean_distances
counts = vectorizer.fit_transform(corpus).todense()
counts


# In[26]:


print('Distance between 1st and 2nd documents:', euclidean_distances(counts[0], counts[1]))
print('Distance between 2nd and 3rd documents:', euclidean_distances(counts[1], counts[2]))
print('Distance between 3rd and 1st documents:', euclidean_distances(counts[2], counts[0]))


# - **Sparse vectors** ::= High-dimensional feature vectors that have many zero-values elements
#     - We want to reduce the dimension because higher dimension data require more space and more trainings

# ## Extending bag-of-words wit TF-IDF weights
# - Sometimes containing a word for a single time can be entirely different than containing a word many times. We add a variable to count for the time a word appears in a text. 

# In[27]:


from sklearn.feature_extraction.text import CountVectorizer
corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
vectorizer = CountVectorizer(stop_words='english')
result = vectorizer.fit_transform(corpus).todense()
print (result,"\n", vectorizer.vocabulary_)


## Now to consider also abouy=t the document frequency
## we introduce the inverse document frequency (IDF), which 
# is a measure of how frequent a word is in a corpus.
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
'The dog ate a sandwich and I ate a sandwich',
'The wizard transfigured a sandwich'
]
vectorizer = TfidfVectorizer(stop_words='english')
result = vectorizer.fit_transform(corpus).todense()
print ("\n\n",result,"\n", vectorizer.vocabulary_)


# #### We can see that words appear more times in the corpus are penalized (e.g., sandwich).

# ### Stop-word filtering
# - Dterminers, auxiliary verbs, prepositions etc..
# - Contribute through grammar rather than denotations
# 

# In[28]:


vectorizer = CountVectorizer(stop_words = "english")
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)


# #### We can see that the vocabulary list ignored stop words "the" and "in".

# ### Stemming
# - A strategy to condense inflected and derived forms of a word into single feature

# In[29]:


vectorizer = CountVectorizer(binary=True, stop_words='english')
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him',
    'I eat that sandwich'
]

counts = vectorizer.fit_transform(corpus).todense()
print(counts)
print(vectorizer.vocabulary_)

print('Distance between 1st and 2nd documents:', euclidean_distances(counts[0], counts[1]))
print('Distance between 2nd and 3rd documents:', euclidean_distances(counts[1], counts[2]))
print('Distance between 3rd and 1st documents:', euclidean_distances(counts[2], counts[0]))


# ### Lemmatization
# - the process of determining the lemma, aka the morphological roots of words
# - stemming remmoves all patterns of characters that appear to be affixes. Might result in generating tokens that are not words

# In[30]:


corpus = [
       'I am gathering ingredients for the sandwich.',
       'There were many wizards at the gathering.'
]


# In[31]:


import nltk
## The toolkit can be used after calling nltk.download()
## But I have no space left on my device for downloading.
## Check p.58 in Mastering Machine Learning with Scikit-Learn.


# ## Space-efficient feature vectorizing with the hashing trick
# - We want to avoid using the dictionary in the previous example.
#     1. Needs one additional loop to create the dictionary 
#     2. Needs memory to store the dictionary
# - Hashing trick
#     - stateless
#     - Since we have hash collision problems, some values are designed to be negative (using a signed function) to cancel errors out rather than to add them up.

# In[37]:


from sklearn.feature_extraction.text import HashingVectorizer
corpus =  ['the', 'ate',"ate", 'bacon', 'cat', "cat", "cat"]
##  n_features is an optional keyword argument. 
## Its default value, 2^20 , is adequate for most problems; 
## it is set to 6 here so that the matrix will be small
## enough to print and still display all of the nonzero features
vectorizer = HashingVectorizer(n_features=6,stop_words = "english")
result = vectorizer.transform(corpus).todense()
print (result)


# #### As what we would expect, 'cat' appeared 3 times, 'ate' 2 times, and 'bacon' 1 time.

# ## Data standardization

# In[53]:


from sklearn import preprocessing
import numpy as np
X =  np.array([
    [0., 0., 5., 13., 9., 1.],
    [0., 0., 13., 15., 10., 15.],
    [0., 3., 15., 2., 0., 11.]
    ])

standardized_X = preprocessing.scale(X)
print (standardized_X)


# In[ ]:




