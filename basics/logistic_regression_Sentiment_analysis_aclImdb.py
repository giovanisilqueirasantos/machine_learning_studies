from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd

"""
    to use stopwords you will need to dowload it
    go to python3 shell and type this:
    >> import nltk
    >> nltk.download('stopwords')
"""

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

df = pd.read_csv('./datasets/aclImdb_sentiment_dataset.csv')

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
stop = stopwords.words('english')

param_grid = [{
    'vect__ngram_range': [(1,1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer, tokenizer_porter],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]
}, {
    'vect__ngram_range': [(1,1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer, tokenizer_porter],
    'vect__use_idf':[False],
    'vect__norm':[None],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]
}]

lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)

print(f'CV Accuracy: {gs_lr_tfidf.best_score_:.3f}')
clf = gs_lr_tfidf.best_estimator_
print(f'Test Accuracy: {clf.score(X_test, y_test):.3f}')
print(gs_lr_tfidf.predict(['this movies is really bad']))