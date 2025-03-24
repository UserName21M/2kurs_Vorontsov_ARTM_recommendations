# %%

from datasets import load_dataset

data = load_dataset("mlsa-iai-msu-lab/ru_sci_bench", split = 'ru')
len(data['title'])

# %%

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('mlsa-iai-msu-lab/sci-rus-tiny')
embeddings = model.encode(['привет мир'])
print(embeddings[0].shape) # (312,)

# %%

data = data.map(
    lambda example: {
        'embedding': model.encode([
            example['title'] + model.tokenizer.sep_token + example['abstract']
        ]).squeeze()
    }
)

# %%

data.save_to_disk('scibench_dataset')

# %%

from datasets import load_from_disk
data = load_from_disk('scibench_dataset')

# %%

list((i, w) for i, w in  enumerate(data['title']) if 'нейрос' in w.lower())

# %%

import numpy as np

target = 184228 # omni-robot
target = 1144 # neural predict time-series

embeddings = np.array(data['embedding'])
size = embeddings.shape[0]
target_emb = embeddings[target]

embeddings.shape

# %%

distance = np.linalg.norm(embeddings - target_emb, ord = 2, axis = 1)
# distance[target] = distance.max() + 1
closest = np.argsort(distance)[:10000]

# %%

titles = data['title']
abstracts = data['abstract']
mydata = [(titles[i], abstracts[i], distance[i]) for i in closest]

# %%

import pandas as pd

df = pd.DataFrame(mydata, columns = ['title', 'abstract', 'distance'])
df.to_csv('scidata.csv', encoding = 'utf-8')

# %%

df.head()

# %%
