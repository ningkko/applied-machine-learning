#!/usr/bin/env python
# coding: utf-8

# 1. Tutorial: [Train an Image Classifier with TensorFlow for Poets - Machine Learning Recipes #6](https://www.youtube.com/watch?v=LDRbO9a6XPU&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=8)
# 2. Read more about decision trees: [A useful view of decision trees](https://www.benkuhn.net/tree-imp)

# ## Decision Tree using CART
# - CART: Classification and Regression Trees
#     - Start with a node containing all the examples
#     - For all possible questions and threshold:
#         Ask a question with a threshold and Split the data
#         Calculate the information gain (which is calculated using the Gini-impurity factor).
#     - Choose the set of question and threshold with the highest information gain and least Gini impurity.
#     - Continue doing so till only one example is left in each node.

# First we define two functions for rounding our data. We round the data because 10 digital numbers can produce too many differences, which makes it too hard to gain info from each question we ask.
# 

# In[3]:


def round_to_nearest_zero_point_five(inp):
    return (inp*2).round()/2

def round_to_nearest_zero_point_two_five(inp):
    return (inp*4).round()/4


# We will see that the more precise the data is, the more accurate the prediction is, but also the heavier the cost (running time) is.

# #### - The data we will use is Iris. First we concatenate X and y.

# In[4]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
import numpy as np

iris=datasets.load_iris()
X = iris.data
y = iris.target.reshape(1,150)

data = np.concatenate((X, y.T), axis=1)


# In[5]:


def split_data(data):
    
    data_train, data_test = train_test_split(data, test_size=0.25, random_state=33)
    print("Training data:\n", data_train)
    print("Testing data:\n", data_test)
    return data_train, data_test


# In[6]:


data_train, data_test = split_data(data.round())


# #### Header contains the content of questions we are going to ask for splitting data

# In[7]:


header = iris.feature_names
header


# #### So the question we will ask is something like "Is the sepal length (cm) >= 3?" and "Is the petal width (cm) >= 2?", with 3 and 2 here as the "threshold values."

# In[8]:


class Question:
    """
    The question class: all pairs of (j, t)
    """
    
    def __init__(self, question_number, threshold_value):
        self.question_number = question_number
        self.threshold_value = threshold_value

    def match(self, example):
        """
        returns true if the example's value to the question is greater or euqal to 
        the threshold value of the question
        """
        value = example[self.question_number]
        return value >= self.threshold_value

    def __repr__(self):
        """
        Prints the process and result ina human-rigidable way
        """
        condition = ">="
        return "Is %s %s %s?" % (
            header[self.question_number], condition, str(self.threshold_value))


# In[9]:


Question(1, 2)


# In[10]:


def partition(dataset, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_example, false_example = [], []
    for row in dataset:
        if question.match(row):
            true_example.append(row)
        else:
            false_example.append(row)
    return true_example, false_example


# In[11]:


def count_number_of_labels(dataset):
    """
    Counts for how many different labels the dataset has.
    """
    counts = {}  
    for data in dataset:
        # Label is always the last one (since we define it this way)
        label = data[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# In[12]:


def gini(dataset):
    """Calculates the Gini Impurity of a dataset.
    """

    counts = count_number_of_labels(dataset)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(dataset))
        impurity -= prob_of_label**2
    return impurity


# In[13]:


## test
gini([[1],[2],[2],[1],[3]])


# In[14]:


def information_gain(left_child, right_child, current_uncertainty):
    """
    Information Gain = parent_node.uncertainty - (left_child.gini_impurity + right_child.gini_impurity)
    """
    left_child_percentage = float(len(left_child)) / (len(left_child) + len(right_child))
    right_child_percentage = 1 - left_child_percentage
    return current_uncertainty - left_child_percentage * gini(left_child) - right_child_percentage * gini(right_child)


# In[15]:


def find_best_split(dataset):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0 
    best_question = None
    current_uncertainty = gini(dataset)
    
    feature_number = len(dataset[0]) - 1  

    ## now for each pair of (j, t), which means (question and threshold), try
    ## splitting the set and calculating the information gain.
    
    for question_number in range(feature_number):
        
        thresholds = set([row[question_number] for row in dataset])
        
        for threshold in thresholds:  
            
            question = Question(question_number, threshold)
            true_examples, false_examples = partition(dataset, question)
            
            ## skip if a pair of (j, t) fails in splitting the set
            if len(true_examples) == 0 or len(false_examples) == 0:
                continue

            info_gain = information_gain(true_examples, false_examples, current_uncertainty)

            if info_gain >= best_gain:
                best_gain, best_question = info_gain, question

    return best_gain, best_question


# In[16]:


best_gain, best_question = find_best_split(data_train)
print("Best gain: ", best_gain, 
      "\nBest question: ", best_question)


# In[17]:


class Leaf:
    """
    a leaf is a list of examples
    """

    def __init__(self, dataset):
        self.predictions = count_number_of_labels(dataset)


# In[18]:


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# In[19]:


def build_tree(dataset):

    info_gain, question = find_best_split(dataset)

    ## base case
    if info_gain == 0:
        return Leaf(dataset)
    
    true_examples, false_examples = partition(dataset, question)

    true_branch = build_tree(true_examples)
    false_branch = build_tree(false_examples)

    return Decision_Node(question, true_branch, false_branch)


# In[20]:


def print_tree(node, spacing=""):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + "|" + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


# In[21]:


def classify(data_test,classifier):
    def classify_one(row, node):
        """See the 'rules of recursion' above."""

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node.predictions

        if node.question.match(row):
            return classify_one(row, node.true_branch)
        else:
            return classify_one(row, node.false_branch)
    
    def print_leaf(counts):
        total = sum(counts.values())
        probs = {}
        for labels in counts.keys():
            probs[labels] = str(int(counts[labels] / total * 100)) + "%"
        return probs

    
    for row in data_test:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify_one(row,classifier))))


# In[22]:


decision_tree_classifier = build_tree(data_train)
print_tree(decision_tree_classifier)


# In[23]:


classify(data_test, decision_tree_classifier)


# When we round the data to its nearest integer, we have 13 wrong classifications.
# 
# Now let's round the data into nearest 0.5

# In[30]:


data_train, data_test = split_data(round_to_nearest_zero_point_five(data))


# In[33]:


decision_tree_classifier = build_tree(data_train)
classify(data_test, decision_tree_classifier)


# Only 4 classifications were wrong.
# 
# Let's try rounding to nearest 0.25

# In[34]:


data_train, data_test = split_data(round_to_nearest_zero_point_two_five(data))


# In[35]:


decision_tree_classifier = build_tree(data_train)
classify(data_test, decision_tree_classifier)


# All of the classifications got right.
