#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load the dataset
df = pd.read_csv('train.csv')
df.head(10)


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.isna().sum()


# In[6]:


plt.figure(figsize=(30,20))
sns.countplot(x='genre',data=df)
plt.xlabel('Movie Genre')
plt.ylabel('Count')
plt.title('Genre Plot ')
plt.show()


# In[7]:


df['genre'].value_counts()


# In[8]:


movie_genre = list(df['genre'].unique())
movie_genre.sort()
movie_genre


# In[9]:


mapping = {'action':0,'adventure':1,'comedy':2,'drama':3,'horror':4,'romance':5,'sci-fi':6,'thriller':7,'other':8}
df['genre'] = df['genre'].map(mapping)
df.head()


# In[10]:


df.drop('id',axis=1,inplace=True)


# In[11]:


df.head()


# In[12]:


import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[13]:


df.shape


# In[14]:


corpus = []
porter = PorterStemmer()

for i in range(0,df.shape[0]):
  dialog = re.sub(pattern = '[^a-zA-Z]',repl = " ",string = df['text'][i])

  dialog = dialog.lower()
  words = dialog.split()
  dialog_word = [word for word in words if word not in set(stopwords.words('english'))]

  words = [porter.stem(word) for word in dialog_word]

  dialog = ' '.join(words)

  corpus.append(dialog)


# In[15]:


drama_words = []
for i in list(df[df['genre']==3].index):
  drama_words.append(corpus[i])

action_words = []
for i in list(df[df['genre']==0].index):
  action_words.append(corpus[i])

comedy_words = []
for i in list(df[df['genre']==2].index):
  comedy_words.append(corpus[i])


drama = ''
action = ''
comedy=''
for i in range(0,3):
  drama +=drama_words[i]
  action +=action_words[i]
  comedy+=comedy_words[i]



# In[18]:


from wordcloud import WordCloud
wordcloud1 = WordCloud(background_color='black',width = 3000,height=2000,min_font_size=10,contour_color='blue').generate(drama)
plt.figure(figsize=(15,20))
plt.imshow(wordcloud1)
plt.axis('off')
plt.title('Drama genre')
plt.show()


# In[19]:


wordcloud2= WordCloud(background_color='black',width = 3000,height=2000,min_font_size=10).generate(action)
plt.figure(figsize=(15,20))
plt.imshow(wordcloud2)
plt.axis('off')
plt.title('Action genre')
plt.show()


# In[20]:


wordcloud3 = WordCloud(background_color='black',width = 3000,height=2000,min_font_size=10,contour_color='blue').generate(comedy)
plt.figure(figsize=(15,20))
plt.imshow(wordcloud3)
plt.axis('off')
plt.title('Comedy genre')
plt.show()


# In[21]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000,ngram_range=(1,2))
X = cv.fit_transform(corpus).toarray()
y = df['genre'].values


pickle.dump(cv, open('cv-transform.pkl', 'wb'))


# In[22]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state= 0)


# In[23]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train,y_train)


# In[24]:


nb_predict = classifier.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score,confusion_matrix
score1= accuracy_score(y_test,nb_predict)
print("  Score  ")
print("Accuracy score is: {}%".format(round(score1*100,2)))


# In[26]:


cm = confusion_matrix(y_test,nb_predict)
print(cm)


# In[27]:


plt.figure(figsize=(15,12))
axis_labels = ['action', 'adventure', 'comedy', 'drama', 'horror', 'romance', 'sci-fi', 'thriller','other',]
sns.heatmap(data=cm, annot=True, cmap="Blues", xticklabels=axis_labels, yticklabels=axis_labels)
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Confusion Matrix for Multinomial Naive Bayes Algorithm')
plt.show()


# In[28]:


best_accuracy = 0.0
alpha_val = 0.0
for i in np.arange(0.1,1.1,0.1):
  temp_classifier = MultinomialNB(alpha=i)
  temp_classifier.fit(X_train, y_train)
  temp_y_pred = temp_classifier.predict(X_test)
  score = accuracy_score(y_test, temp_y_pred)
  print("Accuracy score for alpha={} is: {}%".format(round(i,1), round(score*100,2)))
  if score>best_accuracy:
    best_accuracy = score
    alpha_val = i
print('--------------------------------------------')
print('The best accuracy is {}% with alpha value as {}'.format(round(best_accuracy*100, 2), round(alpha_val,1)))


# In[29]:


classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)


# In[30]:


filename = 'movie-genre-classifier.pkl'
pickle.dump(classifier,open(filename,'wb'))


# In[ ]:




