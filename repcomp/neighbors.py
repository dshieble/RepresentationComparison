"""
This file contains an assortment of helper classes and helper methods for the NeighborsComparison
algorithm
"""

from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
from repcomp.exceptions import (
  ImproperParameterSpecificationException,
  MethodNotImplementedException
)


class BruteForceNeighborsTable(object):
  def __init__(self, embeddings):
    self.embeddings = embeddings
    self.neighbor_object = NearestNeighbors(metric="cosine").fit(self.embeddings)

  def get_neighbors(self, item_index, num_neighbors):
    raw_neighbors = self.neighbor_object.kneighbors(
        self.embeddings[item_index][None, ...],
        n_neighbors=num_neighbors + 1, return_distance=False)[0, :]
    return [n for n in raw_neighbors if not n == item_index][:num_neighbors]


class AnnoyNeighborsTable(object):
  def __init__(self, embeddings, ntrees):
    self.ntrees = ntrees
    self.index = AnnoyIndex(embeddings.shape[1], metric='angular')
    for i, embedding in enumerate(embeddings):
      self.index.add_item(i, embedding)
      self.index.build(ntrees)

  def get_neighbors(self, item_index, num_neighbors):
    raw_neighbors = self.index.get_nns_by_item(item_index, num_neighbors + 1)
    return [n for n in raw_neighbors if not n == item_index][:num_neighbors]


def get_neighbors_table(embeddings, method, ntrees=None):
  """
  This is a factory method for cosine distance nearest neighbor methods.
  Args:
    embeddings (ndarray): The embeddings to index
    method (string): The nearest neighbor method to use
    ntrees (int): number of trees for annoy
  Returns:
    Nearest neighbor table
  """
  if method == "annoy":
    if ntrees is None:
      raise ImproperParameterSpecificationException("ntrees must be defined")
    table = AnnoyNeighborsTable(embeddings, ntrees)
  elif method == "brute":
    table = BruteForceNeighborsTable(embeddings)
  else:
    raise MethodNotImplementedException("{} is not an implemented method".format(method))
  return table


def jaccard(neighbors_1, neighbors_2):
  """
  Compute the jaccard index between two sets
  Args:
    neighbors_1 (iterable)
    neighbors_2 (iterable)
  Returns:
    Jaccard index of neighbors_1 and neighbors_2
  """
  neighbors_1_set = set(neighbors_1)
  neighbors_2_set = set(neighbors_2)
  intersection_size = float(len(neighbors_1_set.intersection(neighbors_2_set)))
  union_size = float(len(neighbors_1_set.union(neighbors_2_set)))
  return intersection_size / union_size
