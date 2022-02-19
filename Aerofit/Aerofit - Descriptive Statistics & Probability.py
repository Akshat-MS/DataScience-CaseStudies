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

# In[96]:


#Importing packages
import numpy as np
import pandas as pd

# Importing matplotlib and seaborn for graphs
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Utility Functions - Used during Analysis

# #### Missing Value - Calculator

# In[97]:


def missingValue(df):
    #Identifying Missing data. Already verified above. To be sure again checking.
    total_null = df.isnull().sum().sort_values(ascending = False)
    percent = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)
    print("Total records = ", df.shape[0])

    md = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
    return md


# ### Categorical Variable Analysis 
#   - Bar plot - Frequency of feature in percentage
#   - Pie Chart

# In[98]:


# Frequency of each feature in percentage.
def cat_analysis(df, colnames, nrows=2,mcols=2,width=20,height=30, sortbyindex=False):
    fig , ax = plt.subplots(nrows,mcols,figsize=(width,height))  
    fig.set_facecolor(color = 'white')
    string = "Frequency of "
    rows = 0                          
    for colname in colnames:
        count = (df[colname].value_counts(normalize=True)*100)
        string += colname + ' in (%)'
        if sortbyindex:
                count = count.sort_index()
        count.plot.bar(color=sns.color_palette("crest"),ax=ax[rows][0])
        ax[rows][0].set_ylabel(string, fontsize=14,family = "Comic Sans MS")
        ax[rows][0].set_xlabel(colname, fontsize=14,family = "Comic Sans MS")      
        count.plot.pie(colors = sns.color_palette("crest"),autopct='%0.0f%%',
                       textprops={'fontsize': 14,'family':"Comic Sans MS"},ax=ax[rows][1])        
        string = "Frequency of "
        rows += 1


# ### Function for Outlier detection
#   - Box plot - for checking range of outliers
#   - distplot - For checking skewness

# In[99]:


def outlier_detect(df,colname,nrows=2,mcols=2,width=20,height=15):
    fig , ax = plt.subplots(nrows,mcols,figsize=(width,height))
    fig.set_facecolor("lightgrey")
    rows = 0
    for var in colname:        
        ax[rows][0].set_title("Boxplot for Outlier Detection ", fontweight="bold")
        plt.ylabel(var, fontsize=12,family = "Comic Sans MS")
        sns.boxplot(y = df[var],color='m',ax=ax[rows][0])
        
        # plt.subplot(nrows,mcols,pltcounter+1)
        sns.distplot(df[var],color='m',ax=ax[rows][1])
        ax[rows][1].axvline(df[var].mean(), color='r', linestyle='--', label="Mean")
        ax[rows][1].axvline(df[var].median(), color='g', linestyle='-', label="Median")
        ax[rows][1].axvline(df[var].mode()[0], color='royalblue', linestyle='-', label="Mode")
        ax[rows][1].set_title("Outlier Detection ", fontweight="bold")
        ax[rows][1].legend({'Mean':df[var].mean(),'Median':df[var].median(),'Mode':df[var].mode()})
        rows += 1
    plt.show()


# ### Function for Bi-variante Analysis
#   - Used countplot for the analysis

# In[100]:


def cat_bi_analysis(df,colname,depend_var,nrows=2,mcols=2,width=20,height=15):
    fig , ax = plt.subplots(nrows,mcols,figsize=(width,height))
    sns.set(style='white')
    rows = 0
    string = " based Distribution"
    for var in colname:
        string = var + string
        sns.countplot(data=df,x=depend_var, hue=var, palette="hls",ax=ax[rows][0])
        sns.countplot(data=df, x=var, hue=depend_var, palette="husl",ax=ax[rows][1])
        ax[rows][0].set_title(string, fontweight="bold",fontsize=14,family = "Comic Sans MS")
        ax[rows][1].set_title(string, fontweight="bold",fontsize=14,family = "Comic Sans MS")
        ax[rows][0].set_ylabel('count', fontweight="bold",fontsize=14,family = "Comic Sans MS")
        ax[rows][0].set_xlabel(var,fontweight="bold", fontsize=14,family = "Comic Sans MS")  
        ax[rows][1].set_ylabel('count', fontweight="bold",fontsize=14,family = "Comic Sans MS")
        ax[rows][1].set_xlabel(var,fontweight="bold", fontsize=14,family = "Comic Sans MS") 
        rows += 1
        string = " based Distribution"
    plt.show()


# ###  Function Bi Multi variant Analysis for Numericals variables with Categrical and dependent variable
#   - Used Boxplot 
#   - Point plot

# In[101]:


def num_mult_analysis(df,colname,category,groupby,nrows=2,mcols=2,width=20,height=15):
    fig , ax = plt.subplots(nrows,mcols,figsize=(width,height))
    sns.set(style='white')
    fig.set_facecolor("lightgrey")
    rows = 0
    for var in colname:
        sns.boxplot(x = category,y = var, hue = groupby,data = df,ax=ax[rows][0])
        sns.pointplot(x=df[category],y=df[var],hue=df[groupby],ax=ax[rows][1]) 
        ax[rows][0].set_ylabel(var, fontweight="bold",fontsize=14,family = "Comic Sans MS")
        ax[rows][0].set_xlabel(category,fontweight="bold", fontsize=14,family = "Comic Sans MS")  
        ax[rows][1].set_ylabel(var, fontweight="bold",fontsize=14,family = "Comic Sans MS")
        ax[rows][1].set_xlabel(category,fontweight="bold", fontsize=14,family = "Comic Sans MS") 
        rows += 1
    plt.show()


# In[102]:


aerofit_data = pd.read_csv("./aerofit_treadmill.csv")


# In[103]:


aerofit_data.head()


# In[104]:


aerofit_data.shape


# In[105]:


aerofit_data.columns


# ### Validating Duplicate Records

# In[106]:


aerofit_data = aerofit_data.drop_duplicates()
aerofit_data.shape


# ### Inference
#   - No dupicates records found.

# ### Missing Data Analysis

# In[107]:


missingValue(aerofit_data).head(5)


# ### Inference
#   - No missing value found.

# ### Unique values (counts) for each Feature

# In[108]:


aerofit_data.nunique()


# ### Unique values (names) are checked for each Features

# In[109]:


aerofit_data['Product'].unique()


# In[110]:


aerofit_data['Age'].unique()


# In[111]:


aerofit_data['Gender'].unique()


# In[112]:


aerofit_data['Education'].unique()


# In[113]:


aerofit_data['MaritalStatus'].unique()


# In[114]:


aerofit_data['Usage'].unique()


# In[115]:


aerofit_data['Fitness'].unique()


# In[116]:


aerofit_data['Income'].unique()


# In[117]:


aerofit_data['Miles'].unique()


# ### Inference
#   - No abnormalities were found in the data.

# ### DataType Validation

# In[118]:


aerofit_data.info()


# ### Inference
#   - **Product, Gender and MaritalStatus** are categorial variables. Hence updating the dtype for same.

# In[119]:


aerofit_data['Gender'] = aerofit_data['Gender'].astype("category")


# In[120]:


aerofit_data['Product'] = aerofit_data['Product'].astype("category")


# In[121]:


aerofit_data['MaritalStatus'] = aerofit_data['MaritalStatus'].astype("category")


# ### Analyzing basic statistics about each feature, such as count, min, max, and mean

# In[122]:


aerofit_data.describe()


# ### Inferences
# 
#   - Huge difference in **income for customers** who purchase treadmills. Ranging between USD 29562 to 104581.

# ## Data Preparation

# ### Dervied Columns¶
#   - Added 2 new feature from Age
#     - "AgeCategory" - Teens, 20s, 30s and Above 40s
#     - "AgeGroup" - 14-20 , 20-30, 30-40 & 40-60
#   - Added 1 new categorial feature based on the income
#     - "IncomeSlab" - Low Income, Lower-middle income,Upper-Middle income and High income

# ### Age Category & Age Group

# In[123]:


bins = [14,20,30,40,60]
labels =["Teens","20s","30s","Above 40s"]
aerofit_data['AgeGroup'] = pd.cut(aerofit_data['Age'], bins)
aerofit_data['AgeCategory'] = pd.cut(aerofit_data['Age'], bins,labels=labels)


# In[124]:


aerofit_data.head()


# ### Income Slab

# In[125]:


bins_income = [29000, 35000, 60000, 85000,105000]
labels_income = ['Low Income','Lower-middle income','Upper-Middle income', 'High income']
aerofit_data['IncomeSlab'] = pd.cut(aerofit_data['Income'],bins_income,labels = labels_income)
aerofit_data.head()


# In[126]:


aerofit_data.info()


# ## Univariante Analysis
#   - Numerical Variables
#     - Outlier Detection
#   - Categorial variables 
#     - Product
#     - Gender
#     - MaritalStatus
#     - AgeGroup
#     - AgeCategory
#     - IncomeSlab

# ### Numerical Variables - Outlier detection
#  - Income
#  - Miles

# In[127]:


col_num = [ 'Income', 'Miles']
outlier_detect(aerofit_data,col_num,2,2,14,12)


# ### Inference
#   -  Both Miles and Income have significant outliers based on the above boxblot.
#   -  Also both are "right-skewed distribution" which means the mass of the distribution is concentrated on the left of the figure.
#   - **Majority of Customers** fall within the **USD 45,000 - USD 60,000** range
#   - There are **outliers over USD 85,000**
#   - Only a few of our customers run more than 180 miles per week

# ### Handling outliers

# In[128]:


aerofit_data_v1 = aerofit_data.copy()


# ### Removing outliers for Income Feature

# In[129]:


#Outlier Treatment: Remove top 5% & bottom 1% of the Column Outlier values
Q3 = aerofit_data_v1['Income'].quantile(0.75)
Q1 = aerofit_data_v1['Income'].quantile(0.25)
IQR = Q3-Q1
aerofit_data_v1 = aerofit_data_v1[(aerofit_data_v1['Income'] > Q1 - 1.5*IQR) & (aerofit_data_v1['Income'] < Q3 + 1.5*IQR)]
plt.show()


# ### Removing outliers for the Mile Feature

# In[130]:


#Outlier Treatment: Remove top 5% & bottom 1% of the Column Outlier values
Q3 = aerofit_data_v1['Miles'].quantile(0.75)
Q1 = aerofit_data_v1['Miles'].quantile(0.25)
IQR = Q3-Q1
aerofit_data_v1 = aerofit_data_v1[(aerofit_data_v1['Miles'] > Q1 - 1.5*IQR) & (aerofit_data_v1['Miles'] < Q3 + 1.5*IQR)]
plt.show()


# In[131]:


col_num = [ 'Income', 'Miles']
outlier_detect(aerofit_data_v1,col_num,2,2,14,12)


# In[132]:


aerofit_data_v1.shape


# ### Inferences
#   - It's true that there are outliers, but they may provide many insights for high-end models that can benefit companies more. Therefore, they should not be removed for further analysis.

# ### Categorical variable Uni-variante Analysis

# In[133]:


aerofit_data.columns


# In[134]:


cat_colnames = ['Product', 'Gender', 'MaritalStatus', 'AgeGroup', 'AgeCategory','IncomeSlab','Fitness']
cat_analysis(aerofit_data,cat_colnames,7,2,14,40)


# ### Inferences 
#   - **83%** of treadmills are bought by customers with incomes between USD dollars 35000-60000, and USD dollars 60,000-85000.
#   - **88%** of treadmills are purchased by customers aged 20 to 40.
#   - The treadmills are more likely to be purchased by married people
#   - Model KP281 is the best-selling product
#   - **Customer with fitness level 3** buy major chuck of treadmills. **(54%)**
#   - Breakdown of Products based on customer purchased -
#     - KP281 - **44%**
#     - KP481 - **33%**
#     - KP781 - **22%**

# ## Bi-Variant Analysis 
#   - Categorical variables
#     - Gender
#     - MaritalStatus
#     - AgeGroup
#     - AgeCategory
#     - IncomeSlab

# ### Bivariant analysis for Categorical variables

# In[135]:


col_names = ['Gender', 'MaritalStatus', 'AgeGroup', 'AgeCategory','IncomeSlab','Fitness','Education']
cat_bi_analysis(aerofit_data,col_names,'Product',7,2,20,45)


# ### Inferences 
#   - **Gender**
#     - **KP781 model** is the most popular among males
#     - **KP281** is equally preferred by men and women
#   - **AgeCategory**
#     - The most useful treadmills product for people **over 40s** is the **KP281 & KP781**.However, they buy fewer treadmills.
#   - **Income**
#     - Customer with high income only buy high end model. **(KP781)**
#   - **Fitness Level**
#     - Customers with 5 fitness level prefer using KP781.(High end Model)
#     - With moderate fitness level , customer prefer using KP281.
#   - **Education**
#     - Customer above 20 years education, purchase only **KP781** model.
#     
#   - The other categorical features show no specific trends.

# ### Bivariante Analysis for Numerical variables

# In[136]:


col_num = [ 'Income', 'Miles']
num_mult_analysis(aerofit_data,col_num,"AgeCategory","Product")


# ### Inferences
#  - Customers using KP781 treadmill model runs more miles.

# In[137]:


col_num = [ 'Income', 'Miles']
num_mult_analysis(aerofit_data,col_num,"Education","Product")


# In[138]:


col_num = [ 'Income', 'Miles']
num_mult_analysis(aerofit_data,col_num,"Fitness","Product")


# ### Inferences 
#   - With Fitness level 4 and 5 tend to use High end models and average number of Miles is very high for the customers.

# ### Correlation between different Numerical variables

# In[139]:


sns.pairplot(aerofit_data, hue='Product')
plt.show()


# In[140]:


plt.figure(figsize = (16, 10))
sns.heatmap(aerofit_data.corr(), annot=True, vmin=-1, vmax = 1,cmap="YlGnBu") 
plt.show()


# ### Inferences
#   - **Miles and Fitness** and **Miles and Usage** are highly correlated, which means if a customer's fitness level is high they use more treadmills.
#   - **Income and education** show a strong correlation. High-income and highly educated people prefer high-end models (KP781), as mentioned during Bivariant analysis of Categorical variables.
#   - There is no corelation between **Usage & Age** or **Fitness & Age** which mean Age should not be barrier to use treadmills or specific model of treadmills.

# ### Analysis using Contingency Tables to Calculate Probabilities
# #### (**Marginal Probabilities, Joint Probabilities, Conditional Probabilities**)
# 
#  - Product - Incomeslab
#  - Product - Gender
#  - Product - Fitness
#  - Product - AgeCategory
#  - Product - Marital Status
# 

# In[141]:


aerofit_data.columns


# ### Product - Income

# In[142]:


pd.crosstab(index=aerofit_data['Product'], columns=[aerofit_data['IncomeSlab']],margins=True) 


# #### Percentage of a low-income customer by total no. of customers (Marginal Probability)

# In[143]:


# Summ of the treadmill purchased by Low-income customer by total no. of customers.
round(14/180,2)*100


# #### Percentage of a high-income customer purchasing a treadmill (Marginal Probability)

# In[144]:


# Summ of the treadmill purchased by high income customer by total no. of customers.
round(17/180,2)*100


# #### Percentage of a High-income customer purchasing KP781 treadmill (Joint Probability)

# In[145]:


# Summ of the treadmill with model KP781 purchased by high income customer by total no. of customers.
round(17/180,2)*100


# #### Percentage of customer with high-Income salary buying treadmill given that Product is KP781 (Conditional Probability)

# In[146]:


round(17/17,2)*100


# ### Inference 
#   - Customers having salary more than **USD dollar 85,000 buys only KP781** (high-end Model). 

# ### Product - Gender

# In[147]:


pd.crosstab(index=aerofit_data['Product'], columns=[aerofit_data['Gender']],margins=True) 


# #### Percentage of a Male customer purchasing a treadmill

# In[148]:


prob = round((104/180),2)
pct = round(prob*100,2)
pct


# #### Percentage of a Female customer purchasing KP781 treadmill

# In[149]:


prob = round((7/180),2)
pct = round(prob*100,2)
pct


# #### Percentage of Female customer buying treadmill given that Product is KP281

# In[150]:


prob = round((40/76),2)
pct = round(prob*100,2)
pct


# ### Inference 
#   - Female customer prefer to buy KP281 & KP481
#   - 53% of female tend to purchase treadmill model KP281

# ### Product - Fitness

# In[151]:


pd.crosstab(index=aerofit_data['Product'], columns=[aerofit_data['Fitness']],margins=True) 


# #### Percentage of a customers having fitness level5 are

# In[152]:


prob = round((31/180),2)
pct = round(prob*100,2)
pct


# #### Percentage of a customer with Fitness Level 5 purchasing KP781 treadmill 

# In[153]:


prob = round((29/180),2)
pct = round(prob*100,2)
pct


# #### Percentage of customer with fitness level-5 buying KP781 treadmill given that Product is KP781

# In[154]:


prob = round((29/31),2)
pct = round(prob*100,2)
pct


# ### Inference 
#   - 94% of customers with fitness level 5, purchased KP781

# ### Product - AgeCategory

# In[155]:


pd.crosstab(index=aerofit_data['Product'], columns=[aerofit_data['AgeCategory']],margins=True) 


# In[ ]:





# In[156]:


prob = round((110/180),2)
pct = round(prob*100,2)
pct


# ### Inference
#   - Teen doesnot prefer to buy KP781
#   - 61% of customer with Age group between 20 and 30 purchase treadmills.

# ### Product - Marital Status

# In[157]:


pd.crosstab(index=aerofit_data['Product'], columns=[aerofit_data['MaritalStatus']],margins=True) 


# In[158]:


prob = round((107/180),2)
pct = round(prob*100,2)
pct


# ### Inferences 
#   - 59 percent of customer with maritial Stuatus as Partnered by the treadmills.

# ## Conclusion (Important Observations):
# 
#   - Model **KP281** is the **best-selling product**. **44.0%** of all treadmill **sales go to model KP281.**
#   - The majority of treadmill customers fall within the **USD 45,000 - USD 80,000** income bracket. **83%** of treadmills are bought by individuals with incomes between **USD dollor 35000 and 85000.**
#   - There are only **8%** of customers with **incomes below USD 35000** who buy treadmills.
#   - **88%** of treadmills are purchased by **customers aged 20 to 40.**
#   - **Miles and Fitness** & **Miles and Usage** are highly correlated, which means if a customer's fitness level is high they use more treadmills.
#   - **KP781** is the only model purchased by a customer who has more than **20 years of education and an income of over USD dollor 85,000.**
#   - With **Fitness level 4 and 5,** the customers tend to use **high-end models** and the **average number of miles is above 150 per week** 

# ## Recommendations
#   - **KP281 & KP481** are popular with customers earning **USD 45,000 and USD 60,000** and can be offered by these companies as **affordable models.**
#   - **KP781** should be marketed as a **Premium Model** and marketing it to **high income groups and educational over 20 years** market segments could result in more sales.
#   - Aerofit should conduct **market research** to determine if it can attract customers with **income under USD 35,000 to expand its customer base.**
#   - The **KP781 is a premium model**, so it is ideally suited for **sporty people** who have a high average weekly mileage.

# In[ ]:




