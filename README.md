# repcomp

`repcomp` (short for representation comparison) is a package for comparing trained embedding models. You can use it to compare Deep Neural Networks, Matrix Factorization models, Graph Embeddings, Word Embeddings, etc.

`repcomp` supports the following embedding comparison approaches:

* Nearest Neighbors: Fetch the nearest neighbor set of each entity according to embedding distances, and compare model A's neighbor sets to model B's neighbor sets.
* Canonical Correlation: Treat embedding components as observations of random variables and compute the canonical correlations between model A and model B. 
* Unit Match: Form a unit-to-unit matching between model A's embedding components and model B's embedding components and measure the correlations of the matched units.

You can install repcomp from pip 

```
pip install repcomp
```

A simple example comparing random embeddings:

```python
  from repcomp.comparison import CCAComparison
  import numpy as np

  # Generate random embedding matrices
  num_samples = 100
  num_components = 10
  embedding_1 = np.random.random((num_samples, num_components))
  embedding_2 = embedding_1 + 0.5 * np.random.random((num_samples, num_components))

  # Run the comparison
  comparator = CCAComparison()
  sim = comparator.run_comparison(embedding_1, embedding_2)
  print("The canonical correlation similarity is {}".format(sim["similarity"]))
```

A more involved example comparing word embeddings:

```python
  import gensim.downloader as api
  import numpy as np
  from repcomp.comparison import NeighborsComparison

  # Load word vectors from gensim
  glove_wiki_50 = api.load("glove-wiki-gigaword-50")
  glove_twitter_50 = api.load("glove-twitter-50")

  # Build the embedding matrices over the shared vocabularies
  shared_vocab = set(glove_wiki_50.vocab.keys()).intersection(
    set(glove_twitter_50.vocab.keys()))
  glove_wiki_50_vectors = np.vstack([glove_wiki_50.get_vector(word) for word in shared_vocab])
  glove_twitter_50_vectors = np.vstack([glove_twitter_50.get_vector(word) for word in shared_vocab])

  # Run the comparison
  comparator = NeighborsComparison()
  print("The neighbors similarity between glove-wiki-gigaword-50 and glove-twitter-50 is {}".format(
    comparator.run_comparison(glove_wiki_50_vectors, glove_twitter_50_vectors)["similarity"]))
```
