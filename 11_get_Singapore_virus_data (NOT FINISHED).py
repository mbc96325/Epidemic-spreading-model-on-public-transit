#!/usr/bin/env python
# coding: utf-8


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# In[3]:


url = 'https://www.moh.gov.sg/2019-ncov-wuhan'
dfs = pd.read_html(url, index_col=0)


# In[4]:


for df in dfs:
    display(df.T)


# ### 2.2 Worldwide Data

# In[5]:


get_ipython().run_cell_magic('writefile', 'data/worldwide.csv', 'Date,Mainland China,Other Locations\n20 Jan 2020,   278,   4\n21 Jan 2020,   326,   6\n22 Jan 2020,   547,   8\n23 Jan 2020,   639,  14\n24 Jan 2020,   916,  25\n25 Jan 2020,  2000,  40\n26 Jan 2020,  2700,  57\n27 Jan 2020,  4400,  64\n28 Jan 2020,  6000,  87\n29 Jan 2020,  7700, 105\n30 Jan 2020,  9700, 118\n31 Jan 2020, 11200, 153\n 1 Feb 2020, 14300, 173\n 2 Feb 2020, 17200, 183\n 3 Feb 2020, 19700, 188\n 4 Feb 2020, 23700, 212\n 5 Feb 2020, 27400, 227\n 6 Feb 2020, 30600, 265\n 7 Feb 2020, 34100, 317\n 8 Feb 2020, 36800, 343\n 9 Feb 2020, 39800, 361\n10 Feb 2020, 42300, 457\n11 Feb 2020, 44300, 476')


# ## 3. Data Preprocessing

# In[6]:


dfsg = pd.read_csv('data/singapore.csv')
dfsg['Date'] = pd.to_datetime(dfsg['Date'], dayfirst=True)
dfsg.head()


# In[7]:


dfw = pd.read_csv('data/worldwide.csv')
dfw['Date'] = pd.to_datetime(dfw['Date'], dayfirst=True)
dfw['Total'] = dfw['Mainland China'] + dfw['Other Locations']


# In[8]:


dfw


# ## 4. Plots

# In[9]:


fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7,12))
dfw.plot(x='Date', y='Mainland China', marker='o', ax=ax[0]);
dfw.plot(x='Date', y='Other Locations', marker='o', ax=ax[1]);
dfsg.plot(x='Date', y='Cases', marker='o', ax=ax[2], label='Singapore');
plt.savefig('2019ncov-trends.png')


# In[ ]:
