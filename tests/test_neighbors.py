import unittest
import numpy as np

from repcomp.neighbors import get_neighbors_table, jaccard


class TestNeighbors(unittest.TestCase):

  def test_neighbors(self):
    np.random.seed(0)
    embeddings = np.random.random((1000, 100)) * 10

    annoy_table = get_neighbors_table(embeddings, "annoy", ntrees=500)
    brute_table = get_neighbors_table(embeddings, "brute")

    jaccards = []
    for ind in np.random.permutation(range(embeddings.shape[0]))[:10]:
      annoy_neighbors = annoy_table.get_neighbors(ind, 10)
      brute_neighbors = brute_table.get_neighbors(ind, 10)
      jaccards.append(jaccard(annoy_neighbors, brute_neighbors))
    assert np.mean(jaccards) > 0.3
