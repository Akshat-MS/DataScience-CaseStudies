#!/usr/bin/env python
# coding: utf-8

# # Business Case: Aerofit - Descriptive Statistics & Probability

# ### About Aerofit
# 
# Aerofit is a leading brand in the field of fitness equipment. Aerofit provides a product range including machines such as treadmills, exercise bikes, gym equipment, and fitness accessories to cater to the needs of all categories of people.
# 
# 
# ### Business Problem
# 
# The market research team at AeroFit wants to identify the characteristics of the target audience for each type of treadmill offered by the company, to provide a better recommendation of the treadmills to the new customers. The team decides to investigate whether there are differences across the product with respect to customer characteristics.
# 
# Perform descriptive analytics to create a customer profile for each AeroFit treadmill product by developing appropriate tables and charts.
# For each AeroFit treadmill product, construct two-way contingency tables and compute all conditional and marginal probabilities along with their insights/impact on the business.
# 
# ### Dataset
# 
# The company collected the data on individuals who purchased a treadmill from the AeroFit stores during the prior three months. The dataset has the following features:
# 
# Dataset link: <a href="https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/001/125/original/aerofit_treadmill.csv?1639992749" >Aerofit_treadmill.csv </a>
# 
# 
# |Feature|Possible Values|
# |-------|---------------|
# |Product Purchased |KP281, KP481, or KP781|
# |Age|	In years|
# |Gender|Male/Female|
# |Education|	In years|
# |MaritalStatus|	Single or partnered|
# |Usage|The avg. no. of times customer plans to use the treadmill each week.|
# |Income|Annual income (in $)|
# |Fitness|Self-rated fitness on a 1-to-5 scale (1-poor shape & 5-excellent shape.)|
# |Miles|The avg. no. of miles the customer expects to walk/run each week|
# 
# ### Product Portfolio:
#   - The KP281 is an entry-level treadmill that sells for dollar 1,500
#   - The KP481 is for mid-level runners that sell for dollar 1,750.
#   - The KP781 treadmill is having advanced features that sell for dollar 2,500.
#  
#   

# ### Importing the required libraries or packages for EDA 

# In[18]:


#Importing packages
import numpy as np
import pandas as pd

# Importing matplotlib and seaborn for graphs
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# ### Utility Functions - Used during Analysis

# #### Missing Value - Calculator

# In[19]:


def missingValue(df):
    #Identifying Missing data. Already verified above. To be sure again checking.
    total_null = df.isnull().sum().sort_values(ascending = False)
    percent = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)
    print("Total records = ", df.shape[0])

    md = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
    return md


# In[3]:


aerofit_data = pd.read_csv("./aerofit_treadmill.csv")


# In[4]:


aerofit_data.head()


# In[5]:


aerofit_data.shape


# In[6]:


aerofit_data.columns


# ### Validating Duplicate Records

# In[7]:


aerofit_data = aerofit_data.drop_duplicates()
aerofit_data.shape


# ### Inference
#   - No dupicates records found.

# ### Missing Data Analysis

# In[22]:


missingValue(aerofit_data).head(5)


# ### Inference
#   - No missing value found.

# ### Unique values (counts) for each Feature

# In[23]:


aerofit_data.nunique()


# ### Unique values (names) are checked for each Features

# In[25]:


aerofit_data['Product'].unique()


# In[26]:


aerofit_data['Age'].unique()


# In[27]:


aerofit_data['Gender'].unique()


# In[28]:


aerofit_data['Education'].unique()


# In[29]:


aerofit_data['MaritalStatus'].unique()


# In[30]:


aerofit_data['Usage'].unique()


# In[31]:


aerofit_data['Fitness'].unique()


# In[32]:


aerofit_data['Income'].unique()


# In[33]:


aerofit_data['Miles'].unique()


# ### Inference
#   - No abnormalities were found in the data.

# ### DataType Validation

# In[34]:


aerofit_data.info()


# ### Inference
#   - No problems with the type of data used.

# In[ ]:





# In[35]:


aerofit_data.describe()


# In[ ]:





# ## Data Preparation

# In[ ]:




