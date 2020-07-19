import pandas as pd
import os
import numpy as np
import re

"""
    Download the dataset here http://ai.stanford.edu/~amaas/data/sentiment/
    and extract in the same directory of this file
"""

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ''.join(emoticons).replace('-', '')
    return text

labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()

for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = f'./aclImdb/{s}/{l}'
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile:
                txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)

df.columns = ['review', 'sentiment']

df['review'] = df['review'].apply(preprocessor)

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./aclImdb_sentiment_dataset.csv', index=False)

df = pd.read_csv('./aclImdb_sentiment_dataset.csv')
print(df.head())