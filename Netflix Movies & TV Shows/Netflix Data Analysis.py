#!/usr/bin/env python
# coding: utf-8

# # Business Case: Netflix - Data Exploration and Visualisation

# ![alt text](https://akm-img-a-in.tosshub.com/indiatoday/images/story/202012/Netflix-New-Feature-Audio-Only_1200x768.jpeg?9TmAZq3wvsTH1jXQNlPkiSKJprCtGBAx& "Logo Title Text 1")

# ### Business Problem
# 
#   - Analyze the data and generate insights that could help Netflix decide which type of shows/movies to produce and how to grow the business.
#   
# ### Dataset  -  <a href="https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/000/940/original/netflix.csv" >Netflix Dataset Link</a>
# 
# The dataset provided to you consists of a list of all the TV shows/movies available on Netflix:
# 
#   - **Show_id:** Unique ID for every Movie / Tv Show
#   - **Type:** Identifier - A Movie or TV Show
#   - **Title:** Title of the Movie / Tv Show
#   - **Director:** Director of the Movie
#   - **Cast:** Actors involved in the movie/show
#   - **Country:** Country where the movie/show was produced
#   - **Date_added:** Date it was added on Netflix
#   - **Release_year:** Actual Release year of the movie/show
#   - **Rating:** TV Rating of the movie/show
#   - **Duration:** Total Duration - in minutes or number of seasons
#   - **Listed_in:** Genre
#   - **Description:** The summary description

# # <u> A high level overview of the Neflix Dataset Exploration and Visualization<u>
# 
#   - **Loading and inspecting the Dataset**
#     - Checking Shape of the Dateset
#     - Meaningful Column names
#     - Validating Duplicate Records
#     - Checking Missing values
#     - Unique values (counts) for each Feature
#     - Unique values (names) are checked for Features with a unique value count below 100
#   - **Dataset Preparation**
#     - DataType Validation
#     - Dervied Columns
#   - **Univariante Analysis**
#     - Movies & TV shows - Distribution

# ### Importing the required libraries or packages for EDA 

# In[102]:


#Importing packages
import numpy as np
import pandas as pd

# Importing matplotlib and seaborn for graphs
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Importing Date & Time util modules
from dateutil.parser import parse


# ## <u> Loading and inspecting the Dataset<u>

# ### Loading the csv file

# In[103]:


netflix_data = pd.read_csv("./netflix.csv")


# In[104]:


netflix_data.head()


# ### Checking Shape and Column names

# In[105]:


netflix_data.shape


# In[106]:


netflix_data.columns


# ### To make the column names more meaningful, "listed_in" has been changed to "genres".

# In[107]:


netflix_data.rename(columns = {"listed_in":"genres"},inplace= True)


# ### Validating Duplicate Records.

# In[108]:


# Dropping Duplicates if any
netflix_data=netflix_data.drop_duplicates()
netflix_data.shape

## No duplicates records found.


# ### Missing Data Anaysis

# In[109]:


#Identifying Missing data. Already verified above. To be sure again checking.
total_null = netflix_data.isnull().sum().sort_values(ascending = False)
percent = ((netflix_data.isnull().sum()/netflix_data.isnull().count())*100).sort_values(ascending = False)
print("Total records (Car Data) = ", netflix_data.shape[0])

missing_data = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
missing_data.head(5)


# ### Inference 
#   -  0.11% of total records have missing data for "date_added". These records can be removed while analyzing the "date_added" feature.
#   - Missing data will be addressed during the analysis of each column.

# ### Unique values (counts) for each Feature

# In[110]:


netflix_data.nunique()


# ### Inference 
#   - dropping show ID as it is just for reference and no use.
#   - Description - 32 movies have the same description. This may be due to movies being released in different languages.

# In[111]:


# dropping show ID as it is just for reference and no use.
netflix_data = netflix_data.drop('show_id',axis=1)


# ### Unique values (names) are checked for Features with a unique value count below 100

# In[112]:


netflix_data['type'].unique()


# In[113]:


netflix_data['rating'].unique()


# In[114]:


netflix_data['release_year'].unique()


# ## <u>Data Preparation<u>

# ### DataType Validation

# In[115]:


netflix_data.info()


# ### Inference 
#   - date_added and release_year is a datetime data type , hence need to update the Dtype

# In[117]:


netflix_data["date_added"] = pd.to_datetime(netflix_data['date_added'])


# In[118]:


netflix_data.info()


# ### Dervied Columns
# 
#   - Added new feature - **"Year Added"** from the **date_added** feature
#   - Added new feature - **"Month Added"** from the **date_added** feature
#   - Added new feature - **"Day Added"** from the **date_added** feature
#   - Added new feature - **"Weekday Added"** from the **date_added** feature

# In[119]:


# Creating the copy of the day before manipulating the date time information.
netflix_date = netflix_data.copy()


# In[125]:


netflix_date.shape


# ### Removed the missing values before Analysising

# In[152]:


netflix_date.dropna(subset = ['date_added'],inplace= True)


# In[153]:


netflix_date.shape


# In[154]:


netflix_date["year_added"] = netflix_date['date_added'].dt.year
netflix_date["year_added"] = netflix_date["year_added"].astype("Int64")
netflix_date["month_added"] = netflix_date['date_added'].dt.month
netflix_date["month_added"] = netflix_date["month_added"].astype("Int64")
netflix_date["day_added"] = netflix_date['date_added'].dt.day
netflix_date["day_added"] = netflix_date["day_added"].astype("Int64")
netflix_date['Weekday_added'] = netflix_date['date_added'].apply(lambda x: parse(str(x)).strftime("%A"))


# In[155]:


netflix_date.info()


# ### Analyzing basic statistics about each feature, such as count, min, max, and mean.

# In[157]:


netflix_date.describe()


# ### Inference
#   - Netflix has **25%** of movies and TV shows that were released within the **last two years**
#   - About **75%** of Netflix's content consists of movies and TV shows **released after 2013**
#   - Data from Netflix shows that **new trend movies or TV shows are more prevalent**.
#   - For more subscribers, Netflix should invest in **classic Movies and TV shows.**

# ## <u>Univariante Analysis<u>
# 

# #### Feature Name
#   - **Type** - Movies & TV shows - Distribution
#   - **date_added** - Checking number of new Contents added yearly, monthly, which date and Weekend-Weekday
#   - **release_year** - 

# ### Movies & TV shows - Distribution

# In[213]:


data = netflix_data.groupby("type")['type'].count()

explode=(0.08,0)
labels = ['Movie', 'TV Show']
colors = sns.color_palette("Reds")
plt.pie(data, labels=labels,colors = colors, autopct = '%0.0f%%', explode = explode)
plt.show()


# ### Inference
#   - Netflix has **70%** of its content as movies
#   - **Movies** are clearly more **popular on Netflix than TV shows**.
# 

# ### Checking number of new Contents added yearly

# In[220]:


fig = plt.figure(figsize=(15,5))
sns.set(style='whitegrid')
fig.set_facecolor("lightgrey")
count = (netflix_date['year_added'].value_counts(normalize=True)*100)
count.plot.bar(color=sns.color_palette('Reds'))
plt.ylabel('"Frequency of Movies added in %', fontsize=12)
plt.xlabel('Year on which Movies added on Netflix', fontsize=12)


# In[221]:


fig = plt.figure(figsize=(15,5))
sns.set(style='whitegrid')
fig.set_facecolor("lightgrey")
sns.countplot(data=netflix_date,x = 'year_added',palette ="Reds")


# In[209]:


fig = plt.figure(figsize=(15,5))
sns.set(style='whitegrid')
fig.set_facecolor("lightgrey")
sns.histplot(data=netflix_date,x = 'year_added',bins = 12,color='r', shrink=.9)


# ### Inference 
#   - 

# ### Checking number of new Contents added montly

# In[208]:


fig = plt.figure(figsize=(15,5))
sns.set(style='whitegrid')
fig.set_facecolor("lightgrey")
sns.countplot(data=netflix_date,x = 'month_added',palette ="Reds")


# ### Inference

# ### Checking number of new Contents on Weekends

# In[222]:


fig = plt.figure(figsize=(15,5))
sns.set(style='whitegrid')
fig.set_facecolor("lightgrey")
count = (netflix_date['Weekday_added'].value_counts(normalize=True)*100)
count.plot.bar(color=sns.color_palette('Reds'))
plt.ylabel('"Frequency of Movies added in %', fontsize=12)
plt.xlabel('Year on which Movies added on Netflix', fontsize=12)


# In[207]:


fig = plt.figure(figsize=(15,5))
sns.set(style='whitegrid')
fig.set_facecolor("lightgrey")
sns.countplot(data=netflix_date,x = 'Weekday_added',palette ="Reds")


# ### Inference

# In[206]:


fig = plt.figure(figsize=(15,5))
sns.set(style='whitegrid')
fig.set_facecolor("lightgrey")
count = (netflix_date['day_added'].value_counts(normalize=True)*100)
count.plot.bar(color=sns.color_palette('Reds'))
plt.ylabel('"Frequency of Movies added in %', fontsize=12)
plt.xlabel('Date on which Movies added on Netflix', fontsize=12)


# In[ ]:





# In[ ]:





# In[217]:


fig = plt.figure(figsize=(15,5))
sns.set(style='whitegrid')
fig.set_facecolor("lightgrey")
sns.countplot(data=netflix_date,x = 'release_year',palette ="Reds")


# In[216]:


fig = plt.figure(figsize=(15,5))
sns.set(style='whitegrid')
fig.set_facecolor("lightgrey")
sns.histplot(data=netflix_date,x = 'release_year',color='r', shrink=.9)


# In[198]:


fig = plt.figure(figsize=(15,5))
sns.set(style='white')
fig.set_facecolor("lightgrey")
sns.countplot(data=netflix_date,x = 'day_added',palette ="Reds")


# In[ ]:


### Inferences
 - 


# In[197]:


fig = plt.figure(figsize=(15,5))
sns.set(style='white')
fig.set_facecolor("lightgrey")
sns.histplot(data=netflix_date,x = 'day_added',bins = 31,kde=True,color='r', shrink=.9)


# In[ ]:


fig = plt.figure(figsize=(15,5))
sns.set(style='white')
fig.set_facecolor("lightgrey")


# In[167]:





# In[178]:


sns.histplot(data=netflix_date,x = 'month_added',bins = 12,kde=True,hue="type",multiple="dodge", shrink=.8)


# In[92]:


fig = plt.figure(figsize=(40,100))
sns.set(style='white')
fig.set_facecolor("lightgrey")

plt.figure(figsize=(20, 12))
plt.subplot(2,2,1)
sns.scatterplot(data=netflix_date, x='CPU_Usg_mhz', y='Mem_Usg_kb',hue="Weekday")
plt.subplot(2,2,2)
sns.scatterplot(data=netflix_date, x='CPU_Usg_mhz', y='disk_r_tp_kb',hue="Weekday")


# In[93]:


netflix_data.info()


# In[95]:


netflix_data.head()


# In[ ]:


netflix_data = netflix_data['date_added'].dt.year


# In[99]:


netflix_date = netflix_data[['date_added']].dropna()


# In[100]:


netflix_date['Weekday'] = netflix_date['date_added'].apply(lambda x: parse(str(x)).strftime("%A"))


# In[101]:


netflix_date.head()


# ## Inferences 
#   - 

# ## Inferences 
#   - We see 40% more **Movies** than **TV Shows** on Netflix.

# In[24]:


netflix_data_v1[['genre_1','genre_2','genre_3']] = netflix_data['genres'].str.split(',', expand=True)


# In[52]:


netflix_data_v1['genre_1'].unique()


# In[21]:


netflix_data_v1.head()


# In[28]:


netflix_data.head()


# In[14]:


netflix_data['cast'].str.split(',', expand=True)


# In[17]:


netflix_data_v1 = netflix_data['listed_in'].str.split(',', expand=True)


# In[19]:


netflix_data_v1.iloc[:,0].value_counts()


# In[23]:


def to_1D(series):
    return pd.Series([x for _list in series for x in _list])


# In[24]:


unique_items = to_1D(netflix_data["listed_in"]).value_counts()


# In[25]:


unique_items


# In[20]:


def boolean_df(item_lists, unique_items):
# Create empty dict
    bool_dict = {}
    
    # Loop through all the tags
    for i, item in enumerate(unique_items):
        
        # Apply boolean mask
        bool_dict[item] = item_lists.apply(lambda x: item in x)
            
    # Return the results as a dataframe
    return pd.DataFrame(bool_dict)


# In[22]:


netflix_data_v2 = boolean_df(netflix_data["listed_in"], unique_items.keys())


# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


netflix_data_v1 = netflix_data["listed_in"].apply(pd.Series)


# In[12]:


netflix_data_v1.head()


# In[ ]:


def clean_alt_list(list_):
    list_ = list_.replace(', ', '","')
    list_ = list_.replace('[', '["')
    list_ = list_.replace(']', '"]')
    return list_


# In[30]:


s = "Docuseries, Reality TV"
l1 = list(lambda s : s.split(","))


# In[29]:


l1


# In[ ]:




