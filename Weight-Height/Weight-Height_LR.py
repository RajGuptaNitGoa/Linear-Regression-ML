#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[15]:


# to show graphs
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


df= pd.read_csv('weight-height.csv')


# In[17]:


df.head()


# In[22]:


## scatter plot
plt.scatter(df['Weight'],df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")


# In[23]:


## correlation
# show the correlation between x and y 
df.corr()
#the value like 0.9.... shows that weight and height  are highly correlated to each other


# In[25]:


## seaborn for visulizaton
import seaborn as sns
sns.pairplot(df)
## display the same coorelation in graphs


# In[31]:


## independent and dependednt features
x=df[['Weight']] ### independent festures should be data frame or 2 dimensional array
y=df['Height'] ## this can be in series form or in 1 dimensional array


# In[29]:


x_series = df['Weight']
np.array(x_series).shape


# In[32]:


## train test split
from sklearn.model_selection import train_test_split


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=42)


# In[36]:


### standardization
  
from sklearn.preprocessing import StandardScaler


# In[64]:


scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)


# In[65]:


x_test=scaler.transform(x_test)


# In[39]:


## apply linaer regression


# In[41]:


from sklearn.linear_model import LinearRegression


# In[42]:


regression = LinearRegression(n_jobs=-1)


# In[43]:


regression.fit(x_train,y_train)


# In[44]:


print("Coefficient or slope :", regression.coef_)
print("intercept : ",regression.intercept_)


# In[46]:


##plt training data plot best fit line
plt.scatter(x_train, y_train)
plt.plot(x_train, regression.predict(x_train))


# In[47]:


##prediction for test data
y_pred=regression.predict(x_test)


# In[68]:


regression.score(x_test,y_pred)


# In[66]:


regression.score(x_test,y_test)


# In[67]:


regression.score(x_train,y_train)


# In[49]:


##performance matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[51]:


mse=mean_squared_error(y_test, y_pred)
mae= mean_absolute_error(y_test, y_pred)
rmse=np.sqrt(mse)


# In[52]:


print(mse)
print(mae)
print(rmse)


# R square 
# formula 
# r^2=1-ssr/sst

# In[53]:


from sklearn.metrics import r2_score


# In[55]:


score=r2_score(y_test,y_pred)
print(score)


# Adjusted R2=1-[(1-r2)*(n-1)/(n-k-1)]
# where 
# r2: the r2 of the model n: the number of observation k: the number of predictior variables
# 
# 

# In[57]:


#display adjustesd R - squared
1-(1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# In[83]:


#prediction for new data
regression.predict(scaler.transform([[80]]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




