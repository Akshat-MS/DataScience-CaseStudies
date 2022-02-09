#!/usr/bin/env python
# coding: utf-8

# # Business Case: - Descriptive Statistics & Probability

# ### Business Problem
# 
# Market researchers from AdRight are assigned the task of identifying the differences between the treadmill products offered by CardioGood Fitness based on customer characteristics, to provide a better recommendation of the treadmills to the new customers.
# 
# Perform descriptive analytics to create a customer profile for each CardioGood Fitness treadmill product line by developing appropriate tables and charts. For each CardioGood Fitness treadmill product, construct two-way contingency tables and compute all conditional and marginal probabilities along with their insights/impact on the business.
# 
# ### Dataset
# 
# In order to collect data, CardioGoodFitness decides to analyze data from treadmill purchases at its retail stores in the last three months.The team identifies the following customer variables to study:
# 
# Dataset Link: <a href="https://www.kaggle.com/saurav9786/cardiogoodfitness?select=CardioGoodFitness.csv">CardioGoodFitness.csv</a>
# 
# |Feature|Possible Values|
# |-------|---------------|
# |Product Purchased |TM195, TM498, or TM798|
# |Gender|Male/Female|
# |Age|	In years|
# |Education|	In years|
# |MaritalStatus|	Single or partnered|
# |Income|Annual income (in $)|
# |Usage|The avg. no. of times customer plans to use the treadmill each week.|
# |Miles|The avg. no. of miles the customer expects to walk/run each week|
# |Fitness|Self-rated fitness on a 1-to-5 scale (1-poor shape & 5-excellent shape.)|
# 

# ### Importing the required libraries or packages for EDA

# In[60]:


#Importing packages
import numpy as np
import pandas as pd

# Importing matplotlib and seaborn for graphs
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from pandas_profiling import ProfileReport


# ### Utility Functions - Used during Analysis

# #### Missing Value - Calculator

# In[22]:


def missingValue(df):
    #Identifying Missing data. Already verified above. To be sure again checking.
    total_null = df.isnull().sum().sort_values(ascending = False)
    percent = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)
    print("Total records = ", df.shape[0])

    md = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
    return md


# ### Bar plot - Frequency of feature in percentage

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))


# In[69]:


# Frequency of each feature in percentage.
def bar_plot_pct(df, colnames, sortbyindex=False):
    fig,axes = plt.subplots(figure(figsize=(20, 50))
    sns.set(style='whitegrid')    
    fig.set_facecolor("lightgrey")
    string = "Frequency of "
    for colname in colnames:
        plt.subplot(5,2,colnames.index(colname)+1)
        count = (df[colname].value_counts(normalize=True)*100)
        string += colname + ' in (%)'
        if sortbyindex:
                count = count.sort_index()
        count.plot.bar(color=sns.color_palette('dark'))
        plt.xticks(rotation = 70,fontsize=14,family="Comic Sans MS")
        plt.yticks(fontsize=14,family="Comic Sans MS")
        plt.ylabel(string, fontsize=14,family = "Comic Sans MS")
        plt.xlabel(colname, fontsize=14,family = "Comic Sans MS")
        string = "Frequency of "


# In[25]:


treadmill_usg_data = pd.read_csv("./CardioGoodFitness.csv")


# In[26]:


treadmill_usg_data.head()


# In[27]:


treadmill_usg_data.shape


# In[28]:


treadmill_usg_data.columns


# ### Validating Duplicate Records.

# In[29]:


# Dropping Duplicates if any
treadmill_usg_data=treadmill_usg_data.drop_duplicates()
treadmill_usg_data.shape


# ### Inference 
#   - No Duplicate records found.

# ### Missing Value

# In[30]:


missingValue(treadmill_usg_data).head(5)


# ### Inference
#   - No missing value found.

# ### Unique values (counts) for each Feature

# In[34]:


treadmill_usg_data.nunique()


# ### Unique values (names) are checked for each Features

# In[35]:


treadmill_usg_data['Product'].unique()


# In[36]:


treadmill_usg_data['Age'].unique()


# In[37]:


treadmill_usg_data['Gender'].unique()


# In[38]:


treadmill_usg_data['Education'].unique()


# In[39]:


treadmill_usg_data['MaritalStatus'].unique()


# In[40]:


treadmill_usg_data['Usage'].unique()


# In[41]:


treadmill_usg_data['Fitness'].unique()


# In[42]:


treadmill_usg_data['Income'].unique()


# In[43]:


treadmill_usg_data['Miles'].unique()


# ### Inference
#   - No abnormalities were found in the data.

# ## Data Preparation

# ### DataType Validation

# In[31]:


treadmill_usg_data.info()


# ### Inference
#   - No problems with the type of data used.

# ### Dervied Columns
# 
#   - Added 2 new feature from Age
#     - **"Age Category"** - Teens, 20s, 30s and Above 40s
#     - **"Age Group"** - 14-20 , 20-30, 30-40 & 40-60 
# 

# In[ ]:





# In[54]:


bins = [14,20,30,40,60]
labels =["Teens","20s","30s","Above 40s"]
treadmill_usg_data['Age Group'] = pd.cut(treadmill_usg_data['Age'], bins)
treadmill_usg_data['Age Category'] = pd.cut(treadmill_usg_data['Age'], bins,labels=labels)


# In[55]:


treadmill_usg_data.head()


# In[56]:


treadmill_usg_data.info()


# In[68]:


fig = plt.figure(figsize=(15,5))
sns.set(style='whitegrid')
fig.set_facecolor("lightgrey")
count = (treadmill_usg_data['Age Group'].value_counts(normalize=True)*100)
count.plot.bar(color=sns.color_palette('Reds'))
#sns.barplot(x='Age Group',y=treadmill_usg_data.index,data=treadmill_usg_data,palette="Reds")
#sns.countplot(data=treadmill_usg_data,x =treadmill_usg_data.index,palette ="Reds")
plt.title('Count Plot - Movies added to Netflix by year ', fontsize=14)
plt.ylabel('"No. of movies added to Netflix', fontsize=12)
plt.xlabel('Year -> (Movies added to Netflix) ', fontsize=12)


# In[62]:


fig = plt.figure(figsize=(12,5))
sns.set(style = "darkgrid")
fig.set_facecolor("lightgrey")
plt.title('Bar plot - based on release_year of Movies & TV shows', fontsize=12)
plt.ylabel('"Count of Movies & TV shows release by year', fontsize=12)
plt.xlabel('Bin of release year ', fontsize=12)
plt.xticks(rotation = 80,fontsize=12)

plt.show()


# In[ ]:





# In[32]:


treadmill_usg_data.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


ProfileReport(treadmill_usg__data)


# In[ ]:




