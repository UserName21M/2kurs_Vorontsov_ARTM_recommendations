# %%

import pandas as pd
import pymorphy3.cli
import pymorphy3.lang

data = pd.read_csv('scidata.csv')
data.shape

# %%

data.head()

# %%

import pymorphy3
import string
import nltk

lemmatize = lambda str: morph.parse(str)[0].normal_form

morph = pymorphy3.MorphAnalyzer()
charfilter = set(list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя') + list(string.ascii_lowercase) + [' '])
stopwords = set(lemmatize(i) for i in nltk.corpus.stopwords.words('russian'))

count = 0
total = data.shape[0]
print(total)

def process(x : str):
    global count
    count += 1
    if count % 10 == 0:
        print(f'{(count / total * 100):.2f}%', end = '\r')

    x = x.lower().replace('\n', ' ').replace('ё', 'е')
    x = ''.join(i for i in x if i in charfilter)
    x = [lemmatize(i) for i in x.split(' ') if len(i) > 2 and len(i) < 20]
    return list(i for i in x if i not in stopwords)

data['tokens'] = data['abstract'].apply(process)
print()

# %%

from collections import Counter

counter = Counter()
data['tokens'].apply(lambda x: counter.update(x))

# %%

data['tokens'] = data['tokens'].apply(lambda x: list(i for i in x if counter.get(i) >= 10))

# %%

data.to_csv('scidata.csv')

# %%

with open('data_vw', 'w', encoding = 'utf-8') as file:
    file.writelines([('doc_%i ' % i) + ' '.join(s) + '\n' for i,s in enumerate(data['tokens'])])

with open('vocab_vw', 'w', encoding = 'utf-8') as file:
    file.writelines([w + '\n' for w, i in counter.items() if i >= 10])

# %%
