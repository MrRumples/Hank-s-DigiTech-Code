#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
scores = pd.read_csv('test_scores.csv')
scores['posttest'] = scores['posttest'].apply(float)
scores['posttest']
x= scores.drop(columns=['student_id','gender','teaching_method','classroom','school_type','school_setting','n_student','school','lunch'])
y = scores['posttest'].to_numpy()
scores
model = DecisionTreeClassifier()
model.fit(x, y)
predictions = model.predict([[1,2]])


# In[74]:


from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
score = accuracy_score(y_test, predictions)
score


# In[86]:


import seaborn as sns
import matplotlib.pyplot as plt
fgrid: sns.FacetGrid = sns.pairplot(data=scores, hue='teaching_method', vars=[
'pretest', 'posttest'])
fgrid.legend.set_title('Posttest')


# In[ ]:




