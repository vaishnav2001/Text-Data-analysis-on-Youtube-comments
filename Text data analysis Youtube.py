#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df=pd.read_csv(r'F:\vaishnav\Python projects\Yotube analysis\UScomments.csv', error_bad_lines=False)
df.head()


# In[7]:


get_ipython().system('pip install textblob')


# In[8]:


from textblob import TextBlob


# In[9]:


#performing sentiment analysis on comments
polarity=[]
for i in df['comment_text']:
    try:
        polarity.append(TextBlob(i).sentiment.polarity)
    except:
        polarity.append(0)


# In[10]:


df['polarity']=polarity
df.head(20)


# In[11]:


#EDA on poaitive comments
pos_comments=df[df['polarity']==1]
pos_comments.head()


# In[12]:


get_ipython().system('pip install wordcloud')


# In[13]:


from wordcloud import WordCloud,STOPWORDS


# In[67]:


#joining all comments
total_poscomments=' '.join(pos_comments['comment_text'])
len(total_poscomments)


# In[68]:


wordcloud=WordCloud(width=1000, height=500, stopwords=set(STOPWORDS)).generate(total_poscomments)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[16]:


#EDA on negative comments
neg_comments=df[df['polarity']==-1]
neg_comments.head()


# In[17]:


total_comments=' '.join(neg_comments['comment_text'])
len(total_comments)


# In[18]:


wordcloud=WordCloud(width=1000, height=500, stopwords=set(STOPWORDS)).generate(total_comments)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[19]:


#Analyze Trending Tags and Views of Youtube
import re


# In[20]:


df2=pd.read_csv(r'F:\vaishnav\Python projects\Yotube analysis\USvideos.csv', error_bad_lines=False)
df2.head()


# In[21]:


#joining tags
all_tags=(' '.join(df2['tags']))
df2.tags[0]


# In[22]:


#removing special character and extra space
tags=re.sub('[^a-zA-Z]',' ',all_tags)
tags


# In[23]:


tags=re.sub(' +',' ', tags)
tags


# In[24]:


wordcloud=WordCloud(width=1000, height=500, stopwords=set(STOPWORDS)).generate(tags)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[25]:


#regression analysis
sns.regplot(data=df2, x='views', y='likes')
plt.title('Regression plot of views vs likes')


# In[26]:


sns.regplot(data=df2, x='views', y='dislikes')
plt.title('Regression plot of views vs dislikes')


# In[27]:


df_corr=df2[['views', 'likes', 'dislikes']].corr()


# In[28]:


sns.heatmap(df_corr,annot=True)


# In[40]:


#Emoji analysis
get_ipython().system('pip install emoji')
import emoji


# In[35]:


df.dropna(inplace=True)


# In[36]:


df['comment_text'].isna().sum()


# In[41]:


#getting emojis in comments
str=''
for i in df['comment_text']:
    list=[c for c in i if c in emoji.UNICODE_EMOJI]
    for element in list:
        str=str+element


# In[42]:


print(str)


# In[43]:


#Number of distinct emoji 
len(set(str))


# In[45]:


res={i:str.count(i) for i in set(str)}
res


# In[46]:


#sorting emojis
res={k:v for k,v in sorted(res.items(), key=lambda item:item[1])}
res


# In[56]:


keys=[*res.keys()]
values=[*res.values()]
keys[3]


# In[61]:


dict={'char':keys[-20:], 'num':values[-20:]}
df3=pd.DataFrame(dict)
df3


# In[59]:


get_ipython().system('pip install plotly')
import plotly.graph_objs as go
from plotly.offline import iplot


# In[63]:


plot=go.Bar(x=df3['char'],
      y=df3['num'])
iplot([plot])


# In[ ]:





# In[ ]:





# In[ ]:




