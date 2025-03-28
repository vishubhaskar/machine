#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("train_vish.csv")
sample_size=int(0.2 * len(df))
df=df.sample(sample_size)
df


# In[2]:


df.shape


# In[3]:


print("Duplicates",df.duplicated().sum())


# In[4]:


df.drop_duplicates(inplace=True)
print("Duplicates",df.duplicated().sum())


# In[5]:


df=df.rename(columns={"A person on a horse jumps over a broken down airplane.":"source text","A person is at a diner, ordering an omelette.":"plagiarized text","0":"label"})
print(df.columns)


# In[6]:


df.head()


# In[7]:


print(df["label"].value_counts())


# In[8]:


import nltk
nltk.download('popular')
import string
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# In[9]:


print("Null Values",df.isnull().sum())


# In[10]:


df.dropna(inplace=True)
print("Null Values",df.isnull().sum())


# In[11]:


# preprocessing text
def preprocess_text(text):
    # remove punctuation
    text=text.translate(str.maketrans("","",string.punctuation))
    #convert to lower case
    text=text.lower()
    #remove stopwords
    stop_words=set(stopwords.words('english'))
    text=" ".join((word for word in text.split()if word not in stop_words))
    return text
preprocess_text("this is my %#$#!^/ text to use for dummy text")


# In[18]:


df['source text']=df['source text'].apply(preprocess_text)
df['plagiarized text']=df['plagiarized text'].apply(preprocess_text)
df


# In[19]:


tfidf_vectorizer=TfidfVectorizer()
x=tfidf_vectorizer.fit_transform(df['source text'] + " " + df['plagiarized text'])
y=df['label']


# In[20]:


# train test split

x_train , x_test , y_train , y_test=train_test_split(x , y ,test_size=0.2 , random_state=42)


# In[15]:


# training the model (logistic regression)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("accuracy ", accuracy_score(y_test,y_pred))
print("classification report ", classification_report(y_test,y_pred))
print("confusion ", confusion_matrix(y_test,y_pred))


# In[16]:


from sklearn.svm import SVC
model=SVC(kernel='linear', random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("accuracy ", accuracy_score(y_test,y_pred))


# In[28]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("accuracy ", accuracy_score(y_test,y_pred))


# In[23]:


import sys
print(sys.executable)
get_ipython().system('{sys.executable}-m pip install xgboost=1.7.6')


# In[25]:


get_ipython().system('pip install xgboost==1.7.6')


# In[26]:


from xgboost import XGBClassifier


# In[41]:


xgb_model=XGBClassifier(n_estimators=200,max_depth=10,learning_rate=0.1,objective='binary:logistic',random_state=42,n_jobs=-1)
xgb_model.fit(x_train,y_train)
y_pred=xgb_model.predict(x_test)

print("accuracy ", accuracy_score(y_test,y_pred))

import pickle
pickle.dump(xgb_model,open('model.pkl','wb'))


# In[42]:


pickle.dump(tfidf_vectorizer,open('tfidf_vectorizer.pkl', 'wb'))


# In[43]:


model=pickle.load(open('model.pkl' ,'rb'))
tfidf_vectorizer=pickle.load(open('tfidf_vectorizer.pkl','rb'))


# In[49]:


# detection system
def detect(input_text):
    # vectorize text
    vectorized_text= tfidf_vectorizer.transform([input_text])
    # do prediction
    result=model.predict(vectorized_text)
    return "plagiarism detected" if result[0]== 1  else "no plagiarism"
input_text= 'boys look over a bridge on to a lake'
detect(input_text)


# In[52]:


def detect(input_text):
    # vectorize text
    vectorized_text= tfidf_vectorizer.transform([input_text])
    # do prediction
    result=model.predict(vectorized_text)
    return "plagiarism detected" if result[0]== 1  else "no plagiarism"
input_text= 'Researchers detected new type of species in amazon forest'
detect(input_text)


# In[ ]:




