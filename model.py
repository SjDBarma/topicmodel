
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("C:/Users/PRIYANGSHU/Desktop/Industry Assignment 1/Project 2 Data Set/train.csv")
test = pd.read_csv("C:/Users/PRIYANGSHU/Desktop/Industry Assignment 1/Project 2 Data Set/test.csv")

print('Train shape:',train.shape)
print('Test shape:',test.shape)
l = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

test = test.drop(['ID'],axis=1)

X = train.loc[:,['TITLE','ABSTRACT']]
y = train.loc[:,l]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

y_test.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)

y1 = np.array(y_train)
y2 = np.array(y_test)

#Removing Punctuations

X_train.replace('[^a-zA-Z]',' ', regex=True, inplace=True)
X_test.replace('[^a-zA-Z]',' ', regex=True, inplace=True)

test.replace('[^a-zA-Z]',' ', regex=True, inplace=True)

#Converting to lower case characters

for index in X_train.columns:
  X_train[index] = X_train[index].str.lower()

for index in X_test.columns:
  X_test[index] = X_test[index].str.lower()

for index in test.columns:
  test[index] = test[index].str.lower()

#Removing one letter words

X_train['ABSTRACT'] = X_train['ABSTRACT'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
X_test['ABSTRACT'] = X_test['ABSTRACT'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')

test['ABSTRACT'] = test['ABSTRACT'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')

#Removing multiple blank spaces

X_train = X_train.replace('\s+', ' ', regex=True)
X_test = X_test.replace('\s+', ' ', regex=True)

test = test.replace('\s+', ' ', regex=True)
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 
# len(stop_words)
# X_train['ABSTRACT'] = X_train['ABSTRACT'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
# X_test['ABSTRACT'] = X_test['ABSTRACT'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

# test['ABSTRACT'] = test['ABSTRACT'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

X_train['combined'] = X_train['TITLE']+' '+X_train['ABSTRACT']
X_test['combined'] = X_test['TITLE']+' '+X_test['ABSTRACT']

test['combined'] = test['TITLE']+' '+test['ABSTRACT']

X_train = X_train.drop(['TITLE','ABSTRACT'],axis=1)
X_test = X_test.drop(['TITLE','ABSTRACT'],axis=1)

test = test.drop(['TITLE','ABSTRACT'],axis=1)

X_train.head()

X_lines = []
for row in range(0,X.shape[0]):
  X_lines.append(' '.join(str(x) for x in X.iloc[row,:]))
     

train_lines = []
for row in range(0,X_train.shape[0]):
  train_lines.append(' '.join(str(x) for x in X_train.iloc[row,:]))

test_lines = []
for row in range(0,X_test.shape[0]):
  test_lines.append(' '.join(str(x) for x in X_test.iloc[row,:]))

predtest_lines = []
for row in range(0,test.shape[0]):
  predtest_lines.append(' '.join(str(x) for x in test.iloc[row,:]))
     

len(train_lines)
from sklearn.feature_extraction.text import CountVectorizer

countvector = CountVectorizer(ngram_range=(1,2))
X_train_cv = countvector.fit_transform(train_lines)
X_test_cv = countvector.transform(test_lines)

test_cv = countvector.transform(predtest_lines)

#Using TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

tfidfvector = TfidfTransformer()
X_train_tf = tfidfvector.fit_transform(X_train_cv)
X_test_tf = tfidfvector.fit_transform(X_test_cv)

test_tf = tfidfvector.fit_transform(test_cv)

X_cv = countvector.transform(X_lines)

X_tf = tfidfvector.fit_transform(X_cv) #x_tf,y
     

from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier

model = LinearSVC(C=0.5, class_weight='balanced', random_state=42)
models = MultiOutputClassifier(model)

models.fit(X_train_tf, y1)

preds = models.predict(X_test_tf)
preds

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#print(confusion_matrix(y2,preds))
print(classification_report(y2,preds))
print(accuracy_score(y2,preds))
predssv = models.predict(test_tf)
predssv

test = pd.read_csv("C:/Users/PRIYANGSHU/Desktop/Industry Assignment 1/Project 2 Data Set/test.csv")

submit = pd.DataFrame({'ID': test.ID, 'Computer Science': predssv[:,0],'Physics':predssv[:,1],
                       'Mathematics':predssv[:,2],'Statistics':predssv[:,3],'Quantitative Biology':predssv[:,4],
                       'Quantitative Finance':predssv[:,5]})
submit.head()

submit.to_csv('submission.csv', index=False)

import pickle
# Save the trained MultiOutputClassifier model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(models, file) 
# Loading the MultiOutputClassifier model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
