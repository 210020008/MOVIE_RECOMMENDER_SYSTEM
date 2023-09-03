#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#movies = pd.read_csv('/content/tmdb_5000_movies.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')


# In[3]:


#credits = pd.read_csv('/content/tmdb_5000_credits.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies.head(1)


# In[5]:


movies = movies.merge(credits,on='title')
movies.head(1)


# In[6]:


movies['original_language'].value_counts()


# In[7]:


movies.info()


# In[8]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head()


# In[9]:


movies.dropna(inplace=True)


# In[10]:


movies.isnull().sum()


# In[11]:


movies['genres']


# In[12]:


import ast
def convert (obj):
    L=[]
    obj = ast.literal_eval(obj)
    for i in obj:
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)


# In[13]:


movies['genres']


# In[14]:


def convert_cast (obj):
    L=[]
    obj = ast.literal_eval(obj)
    for i in obj:
        if len(L)>=3 :
            break
        L.append(i['name'])
    return L

movies['cast'] = movies['cast'].apply(convert_cast)


# In[15]:


movies['overview']


# In[16]:


def convert_crew (obj):
    L=[]
    obj = ast.literal_eval(obj)
    for i in obj:
        if i['job'] == "Director":
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(convert_crew)


# In[17]:


movies['crew']


# In[18]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[19]:


movies['overview']


# In[20]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[21]:


movies.iloc[0]


# In[22]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[23]:


movies.iloc[0]


# In[24]:


new_df = movies[['movie_id','title','tags']]
new_df


# In[25]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[26]:


new_df


# In[27]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem (text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)


# In[28]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[29]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000 , stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()


# In[30]:


cv.get_feature_names_out()


# In[31]:


vectors


# In[32]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)


# In[33]:


def recommend_1 (movie):
    index = 0
    for i in new_df['title']:
        if i == movie:
            break
        index += 1
    row_to_sort = similarity[index]
    sorted_indices = np.argsort(row_to_sort)[::-1]
    count = 1
    l = []
    for i in sorted_indices:
        if count == 7:
            break
        if count != 1:
            l.append(i)
        count += 1
    names_l=[]
    count = 0
    for i in new_df['title']:
        if len(names_l)>=5:
            break
        if count == l[0] or count == l[1] or count == l[2] or count == l[3] or count == l[4]:
            names_l.append(i)
        count += 1
    return names_l

recommend_1('Avatar')


# In[38]:


def recommend (movie):
    movie_index=new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df[iloc[0]].title)


# In[ ]:





# In[ ]:




