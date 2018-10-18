# from collections import namedtuple
import logging
import time

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from repcomp.exceptions import EmbeddingShapeMismatchException
from repcomp.neighbors import get_neighbors_table, jaccard


class Comparison(object):

  def __init__(self, pca_components=None):
    self.pca_components = pca_components

  def _compare(self, embeddings_1, embeddings_2):
    assert False

  def run_comparison(self, embeddings_1, embeddings_2, *args, **kwargs):
    if not embeddings_1.shape[0] == embeddings_2.shape[0]:
      raise EmbeddingShapeMismatchException(
        "embeddings_1 and embeddings_2 must have the same number of elements. " +
        "{} and {} do not match".format(embeddings_1.shape, embeddings_2.shape))
    start = time.time()

    if self.pca_components is not None:
      logging.debug("Performing PCA on the embeddings")
      embeddings_1 = PCA(n_components=self.pca_components).fit_transform(embeddings_1)
      embeddings_2 = PCA(n_components=self.pca_components).fit_transform(embeddings_2)
      logging.debug("PCA Complete")
    out = self._compare(embeddings_1, embeddings_2, *args, **kwargs)
    logging.debug("Computation completed in {} seconds".format(time.time() - start))
    return out


class UnitMatchComparison(Comparison):
  """
  This method compares two embeddings by matching the elements in one embedding to those of
  another embedding
  """

  def __init__(self, replacement=True, metric='correlation', **kwargs):
    self.replacement = replacement
    self.metric = metric
    super(UnitMatchComparison, self).__init__(**kwargs)

  # TODO: investigate and maybe implement cross-val based comparison
  def _compare(self, embeddings_1, embeddings_2):
    if not embeddings_1.shape[1] == embeddings_2.shape[1]:
      raise EmbeddingShapeMismatchException(
        "UnitMatchComparison requires that both embeddings have the same number of units")
    sims = 1 - cdist(embeddings_1.T, embeddings_2.T, metric=self.metric)
    matches = linear_sum_assignment(sims)[0] if not self.replacement else np.argmax(sims, axis=1)
    max_corrs = [sims[i][matches[i]] for i in range(sims.shape[0])]
    return {
      "similarity": np.mean(max_corrs),
      "stdev_similarity": np.std(max_corrs)
    }


class NeighborsComparison(Comparison):
  """
  This method compares two embeddings by comparing the nearest neighbors of the embeddings
  to each other
  """

  def __init__(self, nn_method="annoy", ntrees=500, num_neighbors=10, **kwargs):
    self.nn_method = nn_method
    self.ntrees = ntrees
    self.num_neighbors = num_neighbors
    super(NeighborsComparison, self).__init__(**kwargs)

  def _compare(self, embeddings_1, embeddings_2):
    table_1 = get_neighbors_table(
        embeddings=embeddings_1, method=self.nn_method, ntrees=self.ntrees)
    table_2 = get_neighbors_table(
        embeddings=embeddings_2, method=self.nn_method, ntrees=self.ntrees)

    # For each item, compute the average jaccard index between the nearest neighbor sets
    similarities = []
    for i in range(embeddings_1.shape[0]):
      neighbors_1 = table_1.get_neighbors(i, self.num_neighbors)
      neighbors_2 = table_2.get_neighbors(i, self.num_neighbors)
      similarities.append(jaccard(neighbors_1, neighbors_2))
    return {
      "similarity": np.mean(similarities),
      "stdev_similarity": np.std(similarities)
    }


class CCAComparison(Comparison):
  """
  This method compares two embeddings by computing the CCA between the embedding spaces
  """

  def __init__(self, cca_components=None, **kwargs):
    self.cca_components = cca_components
    super(CCAComparison, self).__init__(**kwargs)

  def _compare(self, embeddings_1, embeddings_2):
    if not embeddings_1.shape[1] == embeddings_2.shape[1]:
      raise EmbeddingShapeMismatchException(
        "CCAComparison requires that both embeddings have the same number of units")
    n_components = embeddings_1.shape[1] if self.cca_components is None else self.cca_components
    cca = CCA(n_components=n_components)
    cca.fit(embeddings_1, embeddings_2)
    X_c, Y_c = cca.transform(embeddings_1, embeddings_2)
    corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
    return {
      "similarity": np.mean(corrs),
      "corrs": corrs
    }
