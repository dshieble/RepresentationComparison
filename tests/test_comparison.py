import unittest
import numpy as np

from repcomp.comparison import NeighborsComparison, UnitMatchComparison, CCAComparison


class TestComparison(unittest.TestCase):

  def test_basic_comparison(self):
    np.random.seed(0)
    transformer = np.random.random((100, 100))
    embeddings_1 = np.random.random((1000, 100))
    embeddings_2 = np.random.random((1000, 100))
    embeddings_3 = np.dot(embeddings_1, transformer) + np.random.random((1000, 100))
    for comparator in [
        NeighborsComparison(nn_method="annoy"),
        NeighborsComparison(nn_method="brute"),
        CCAComparison(pca_components=10),
        CCAComparison(),
        UnitMatchComparison(replacement=True),
        UnitMatchComparison(replacement=False)]:
      s1 = comparator.run_comparison(embeddings_1, embeddings_2)['similarity']
      s2 = comparator.run_comparison(embeddings_1, embeddings_3)['similarity']
      s3 = comparator.run_comparison(embeddings_2, embeddings_3)['similarity']
      assert s2 > s1
      assert s2 > s3

  def test_unitmatch_comparison(self):
    permuter = np.random.permutation(range(100))
    embeddings = np.random.random((1000, 100))
    perm_embeddings = np.vstack([embed[permuter] for embed in embeddings])
    sim = UnitMatchComparison().run_comparison(
      embeddings, perm_embeddings)['similarity']
    assert np.isclose(sim, 1.0)
