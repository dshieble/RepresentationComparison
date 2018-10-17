import time
import logging

import tensorflow as tf
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder

from embeddingcomp.comparison import CCAComparison, NeighborsComparison, UnitMatchComparison
from wals import train_model_on_sparse_matrix


def run_factorization(sparse_matrix, algorithm, latent_factors, unobs_weight):
  """
  Runs WALS or SVD to compute embeddings
  Args:
    sparse_matrix (coo_matrix): ratings matrix
    algorithm (str): svd or wals
    latent_factors (int): The number of embedding factors
    unobs_weight (float): The weight of the unobserved elements
  Returns:
    user_embeddings, item_embeddings
  """
  if algorithm == "wals":
    output_row, output_col = train_model_on_sparse_matrix(
      sparse_matrix, latent_factors=latent_factors, unobs_weight=unobs_weight)
  elif algorithm == "svd":
    U, S, VT = svds(sparse_matrix.asfptype(), k=latent_factors)
    sigma_sqrt = np.diag(np.sqrt(S))
    output_row = np.dot(U, sigma_sqrt)
    output_col = np.dot(sigma_sqrt, VT).T
  else:
    assert False
  return output_row, output_col


def create_matrix(ratings_df):
  """
  Creates sparse matrix from dataframe
  Args:
    ratings_df (dataframe): ratings dataframe
  Returns:
    sparse matrix, ordered list of users, ordered list of items
  """
  user_encoder, item_encoder = LabelEncoder(), LabelEncoder()
  user_encoder.fit(ratings_df['user_id'])
  item_encoder.fit(ratings_df['item_id'])
  user_inds = user_encoder.transform(ratings_df['user_id'].values)
  item_inds = item_encoder.transform(ratings_df['item_id'].values)
  ratings = ratings_df['rating'].values
  sparse_matrix = coo_matrix((ratings, (user_inds, item_inds)),
    shape=(len(user_encoder.classes_), len(item_encoder.classes_)))
  return sparse_matrix, user_encoder.classes_, item_encoder.classes_


def get_embeddings_from_ratings_df(ratings_df, algorithm, latent_factors, unobs_weight):
  """
  Given ratings_df, runs matrix factorization and computes embeddings
  Args:
    ratings_df (dataframe): ratings dataframe
    algorithm (str): svd or wals
    latent_factors (int): The number of embedding factors
    unobs_weight (float): The weight of the unobserved elements
  Returns:
    user_embeddings, item_embeddings
  """
  sparse_matrix, users, items = create_matrix(ratings_df)

  start = time.time()
  tf.reset_default_graph()
  output_row, output_col = run_factorization(sparse_matrix, algorithm, latent_factors, unobs_weight)
  logging.info("time elapsed {}".format(int(time.time() - start)))

  user_to_embedding = {user: output_row[ind] for ind, user in enumerate(users)}
  item_to_embedding = {item: output_col[ind] for ind, item in enumerate(items)}
  return user_to_embedding, item_to_embedding


def compare_embedding_maps(embedding_map_1, embedding_map_2):
  """
  Compares two embedding maps with the similarity comparisons
  Args:
    embedding_map_1 (dict): map from item to embedding
    embedding_map_2 (dict): map from item to embedding
  Returns:
    dictionary with similarities
  """
  shared_items = list(set(embedding_map_1.keys()).intersection(embedding_map_2.keys()))
  embeddings_1 = np.vstack([embedding_map_1[item] for item in shared_items])
  embeddings_2 = np.vstack([embedding_map_2[item] for item in shared_items])
  comparators = [
    ("neighbor", NeighborsComparison()),
    ("cca", CCAComparison()),
    ("unitmatch", UnitMatchComparison())]
  return {name: comparator.run_comparison(embeddings_1, embeddings_2)['similarity']
    for name, comparator in comparators}
