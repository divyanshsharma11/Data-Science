#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PROJECT BY DIVYANSH SHARMA/PRESIDENCY UNIVERSITY
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
pd.core.common.is_list_like = pd.api.types.is_list_like
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import DataReader
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from datetime import datetime


# In[3]:


from __future__ import division


# In[4]:


tech_list = ['AAPL','GOOG','MSFT','AMZN']


# In[5]:


end = datetime.now()
start = datetime(end.year-1,end.month,end.day)


# In[6]:


for stock in tech_list:
    globals()[stock] = DataReader(stock,'yahoo',start,end)


# In[7]:


AAPL.describe()


# In[8]:


AAPL.info()


# In[9]:


#HISTORICAL VIEW OF THE CLOSING PRICE OF AAPL
AAPL['Adj Close'].plot(legend=True,figsize=(10,4))


# In[10]:


AAPL['Volume'].plot(legend=True,figsize=(11,5))


# In[11]:


GOOG.info()


# In[12]:


sns.catplot(x='Adj Close',y='Close',data=AMZN,legend=True,palette='spring')


# In[13]:


ma_day = pd.Series([10,20,300])

for ma in ma_day:
    
    column_name= ("MA for %s days") %(str(ma))
    AAPL[column_name]= ma_day.rolling(ma).mean() 
  


# In[14]:


AAPL[[column_name]].plot(legend=True,figsize=(10,4))


# In[15]:


AAPL['Daily Return']=AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(legend=True,figsize=(10,4),linestyle='--',marker='o')


# In[16]:


sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='blue')


# In[17]:


#Just want adjusted closing point
closing_df = DataReader(tech_list,'yahoo',start,end)['Adj Close'] 


# In[18]:


closing_df.head()


# In[19]:


#PCT_CHANGE = PERCCENTAGE CHANGE
tech_rets = closing_df.pct_change()


# In[20]:


tech_rets.head()


# In[21]:


tech_rets.hist()


# In[22]:


sns.jointplot('AMZN','AMZN',tech_rets,kind = 'scatter',color = 'green')


# In[23]:


sns.jointplot('AMZN','MSFT',tech_rets,kind = 'scatter',color = 'green')


# # USE OF SEABORN AND PANDAS FOR COMPARISON OF STOCKS

# In[24]:


sns.pairplot(tech_rets.dropna())


# In[25]:


# `size` paramter has been renamed to `height`; please update your code.
returns_fig = sns.PairGrid(tech_rets.dropna())
returns_fig.map_lower(plt.scatter,color='green')
returns_fig.map_upper(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=40)


# In[26]:


returns_fig = sns.PairGrid(closing_df)
returns_fig.map_lower(plt.scatter,color='green')
returns_fig.map_upper(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=40)


# In[27]:


#CORREALTION PLOTS  OR HEATMAP AS CORRPLOT HAS DEPRECIATED
#sns.corrplot(tech_rets.dropna(),annot=True)
sns.heatmap(tech_rets.dropna().corr(), annot=True)
plt.show()


# # RISK ANALYSIS

# There are many ways we can quantify risk,one of the most basic ways using thr information we have gather on Daily percentage returns is by comparing the expected return with the standard deviation of the daily returns.

# In[28]:


rets = tech_rets.dropna()


# In[29]:


area = np.pi*10

plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')
#Label the scatter plots,for more info on how this is done, check out the line below
# http://matplotlib.org/users/annotations_guide.html
for label,x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(
        label,
        xy = (x,y),xytext = (50,50),
        textcoords='offset points',ha='right',va='bottom',
        arrowprops=dict(arrowstyle='->', connectionstyle = 'arc3,rad=-0.9'))


# # Value at Risk

# Let's go ahead and define a value at risk parameter for our stocks.We can treat value at risk as the money we could expect to lose(aka putting at risk)for a given confidence interval.These several methods we can use for estimating a value at risk.

# # VALUE AT RISK USING BOOTSTRAP METHOD

# In[30]:


sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='seagreen')


# In[31]:


rets.head()


# In[32]:


rets['AAPL'].quantile(0.05)


# In[33]:


rets['GOOG'].quantile(0.05)


# In[34]:


rets['MSFT'].quantile(0.05)


# In[35]:


rets['AMZN'].quantile(0.05)

