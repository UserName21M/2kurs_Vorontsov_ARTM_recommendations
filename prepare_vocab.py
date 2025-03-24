# %%

import artm

path = 'vocab_tfidf_'
dictionary = artm.Dictionary()
dictionary.gather(data_path = 'batches', vocab_file_path = 'vocab_vw')
dictionary.save_text(path)

# %%

import numpy as np

with open(path, 'r') as file:
    vocab = [i.replace('\n', '') for i in file.readlines()]

with open('data_vw', 'r') as file:
    docs = [i.replace('\n', '').split(' ') for i in file.readlines()]
docsN = len(docs)

with open(path[:-1], 'w') as file:
    file.writelines([i + '\n' for i in vocab[:2]])
    total = len(vocab) - 2
    for i, line in enumerate(vocab[2:]):
        if i % 50 == 0 or i == total - 1:
            print('%.2f%%' % (i / (total - 1) * 100), end = '\r')
        line = line.split(', ')
        word, class_, value, tf, df = line
        df = float(df)
        idf = np.log(docsN / (df + 1))
        tf = 0
        for doc in docs:
            tfd = doc.count(word) / len(doc)
            tf += tfd
#        tf = np.sqrt(tf / docsN)
#        tf = tf / docsN
        tf = 1
        tf_idf = tf * idf
        line[2] = str(tf_idf)
        file.write(', '.join(line) + '\n')
print()

# %%
