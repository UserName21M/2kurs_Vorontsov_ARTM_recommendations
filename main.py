# %%

import pandas as pd

data = pd.read_csv('scidata.csv')
data.shape

# %%

import artm
import artm.score_tracker
import numpy as np

bv = artm.BatchVectorizer(data_path = 'data_vw', data_format = 'vowpal_wabbit',
                          batch_size = 1000, target_folder = 'batches')

dictionary = artm.Dictionary()
# dictionary.gather(data_path = 'batches', vocab_file_path = 'vocab_vw')
dictionary.load_text(dictionary_path = 'vocab_tfidf')

# %%

model = artm.ARTM(num_topics = 100, num_document_passes = 10, dictionary = dictionary)

model.scores.add(artm.PerplexityScore(name = 'perplexity', dictionary = dictionary))
model.scores.add(artm.TopTokensScore(name = 'top-tokens', num_tokens = 10))
model.scores.add(artm.TopicKernelScore(name = 'kernels', probability_mass_threshold = 0.3))
model.scores.add(artm.SparsityPhiScore(name = 'sparsity'))

model.regularizers.add(artm.DecorrelatorPhiRegularizer(name = 'decorrelator', tau = 1e3))
model.regularizers.add(artm.SmoothSparsePhiRegularizer(name = 'idf', tau = 4e-2, dictionary = dictionary))
model.regularizers.add(artm.SmoothSparsePhiRegularizer(name = 'sparse_phi', tau = -0.05))
model.regularizers.add(artm.SmoothSparseThetaRegularizer(name = 'sparse_theta', tau = 0.05))
model.regularizers.add( artm.SmoothSparsePhiRegularizer(name = 'background_regularizer', topic_names = ['topic_0'], tau = 10.0))

was = float('inf')
for i in range(32):
    model.fit_offline(bv)
    perplex = model.score_tracker['perplexity'].last_value
    print(f'Iter #{i}, perplexity: {perplex}, sparsity: {model.score_tracker['sparsity'].last_value}')

    delta = was - perplex
    if 0 <= delta < 0.1 and i >= 10:
        break
    was = perplex

# %%

theta : np.ndarray = model.transform(batch_vectorizer = bv, theta_matrix_type = 'dense_theta').sort_index(axis = 1).to_numpy().T
# theta : np.ndarray = model.get_theta().T.to_numpy()

query : np.ndarray = theta[0]
query[query < 1e-8] = 1e-8
query_norm = query / query.max()
mask = query_norm > 0.6

print(*(query_norm * 100 * mask).astype(np.int32))
print(np.where(mask))

ranks = []
graph = []

print(theta.shape)
for i in range(1, theta.shape[0]):
    doc : np.ndarray = theta[i]
    doc[doc < 1e-8] = 1e-8
    doc_norm = doc / doc.max()

    coverage_themes = np.maximum(query_norm * mask - doc_norm * 2, 0)
    coverage = np.sum(coverage_themes)

    mean = (doc + query) / 2
    similarity = (np.sum(doc * np.log(doc / mean)) + np.sum(query * np.log(query / mean))) / 2
    similarity = np.sqrt(similarity)

    similar_themes = (coverage_themes < 0.3) & mask
    if i < 100:
        graph.append((similarity, coverage, data['distance'][i]))
        print(*(doc_norm * 100).astype(np.int32))
        print(np.where(doc_norm > 0.6))
        print(*graph[-1])

    ranks.append((coverage, similarity, np.sum(similar_themes), np.where(similar_themes), i))

criterion = lambda x: x[1] * 1.0 + x[0] * 1.0 # similarity * k1 + coverage * k2
ranks = list(sorted(ranks, key = criterion)) # minimize
result = ranks[:15]

print(*result, sep = '\n')

# %%

%matplotlib inline
import matplotlib.pyplot as plt

x = [i[2] for i in graph]
y = [i[1] for i in graph]

plt.plot(x, y, 'ro')
plt.show()

# %%

top_tokens = model.score_tracker['top-tokens'].last_tokens

for topic_name in model.topic_names:
    print(topic_name, top_tokens[topic_name])

# %%

kernels = model.score_tracker['kernels'].last_tokens

for topic_name in model.topic_names:
    print(topic_name, kernels[topic_name])

# %%

for doc in result:
    print(data['title'].iloc[doc[-1]]) #, '\t', data['distance'].iloc[doc[-1]], doc[1], doc[0], doc[-1])

# %%

for i, doc in data.iterrows():
    if i > 15:
        break
    print(doc['title'])
#    print(doc['title'], doc['distance'])

# %%

phi = model.get_phi_dense()[0].T
for theme in phi:
    print((theme * theme).sum() * 100)

# %%
