"""Useful functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np
from scipy.linalg import block_diag


def zero_pad(vec, n):
  if len(vec) > n:
    raise ValueError("len(vec) should be less than or equal to n.")
  elif len(vec) == n:
    return vec
  else:
    return np.concatenate([vec, np.zeros(n - len(vec))], axis=0)


def edgelist2graph(edges, remove_floor=True):
  if remove_floor:
    edges = [e for e in edges if 0 not in e]
  return nx.from_edgelist(edges)


def edgelist2adj(edges, remove_floor=True):
  return np.array(nx.to_numpy_matrix(edgelist2graph(edges, remove_floor)))


def adj2graph(adj, **kwargs):
  return nx.from_numpy_matrix(np.matrix(adj), **kwargs)


def adj2edgelist(adj, **kwargs):
  edge_list = nx.to_edgelist(nx.from_numpy_array(adj, **kwargs))
  return [(i, j) for i, j, _ in edge_list]


def simplices_to_adj(simplices, n=None):
  if n is None:
    n = int(np.max(simplices) + 1)
  adj = np.zeros((n, n))
  for s in simplices:
    for i in s:
      for j in s:
        if i != j:
          adj[i, j] = 1
  return adj


def separate_adj(adj, labels):
  return [adj[labels == l][:, labels == l] for l in np.unique(labels)]


def edgelist_vals_to_array(edge_list, vals, n):
  """Arranges list of values over edges as a n x n x vals.shape[1] array."""
  if len(vals.shape) == 1:
    arr = np.zeros((n, n))
  elif len(vals.shape) == 2:
    arr = np.zeros((n, n, vals.shape[1]))
  else:
    raise ValueError("vals cannot have more than 2 dims")

  for v_k, (i, j) in zip(vals, edge_list):
    arr[i, j] = v_k
  return arr


def l1_normalize_rows(mat):
  """Normalize non-zero rows of mat so that they sum to 1.

  Args:
    mat: matrix to normalize

  Returns:
    l1normmat: matrix with rows that sum to 1 or 0.
  """
  denom = np.sum(mat, axis=1)
  denom[denom == 0] = 1.
  l1normmat = np.divide(mat.T, denom).T
  return l1normmat


def get_intersection(pts1, pts2):
  """Get intersection point between two lines.

  Args:
    pts1: two points on line 1
    pts2: two points on line 2

  Returns:
    (x, y) coordinate of intersection if single intersection is found.
      Otherwise returns True (if infinite intersections) or False (if zero).
  """
  xy11, xy12 = pts1
  x11, y11 = xy11
  x12, y12 = xy12
  if x11 != x12:
    m1 = (y12 - y11) * 1. / (x12 - x11)
    b1 = y11 - m1*x11

  xy21, xy22 = pts2
  x21, y21 = xy21
  x22, y22 = xy22
  if x21 != x22:
    m2 = (y22 - y21) * 1. / (x22 - x21)
    b2 = y21 - m2*x21

  if (x11 == x12) and (x21 == x22):
    if x11 == x21:
      return True
    else:
      return False
  elif x11 == x12:
    x_int = x11
    y_int = m2 * x11 + b2
  elif x21 == x22:
    x_int = x21
    y_int = m1 * x21 + b1
  elif m1 == m2:
    if b1 == b2:
      return True
    else:
      return False
  else:
    x_int = (b2 - b1) / (m1 - m2)
    y_int = m1 * x_int + b1

  return (x_int, y_int)


def check_if_intersect(s1, s2, atol=1e-7):
  """Check if two segment structs intersect.

  Args:
    s1: segment 1 with bounds ((x1, y1), (x2, y2))
    s2: segment 2 with bounds ((x1, y1), (x2, y2))
    atol: tolerance for intersection on segment . (default 1e-7)

  Returns:
    do_intersect: True if segments intersect, False otherwise
  """
  xy_intersection = get_intersection(s1, s2)
  if not xy_intersection:
    return False
  if not np.iterable(xy_intersection):
    return False
  x_int, y_int = xy_intersection

  # check if intersection point lies on both segments
  xy11, xy12 = s1
  x11, y11 = xy11
  x12, y12 = xy12
  if not ((np.min([x11, x12])-atol <= x_int <= np.max([x11, x12])+atol) and
          (np.min([y11, y12])-atol <= y_int <= np.max([y11, y12])+atol)):
    return False

  xy21, xy22 = s2
  x21, y21 = xy21
  x22, y22 = xy22
  if not ((np.min([x21, x22])-atol <= x_int <= np.max([x21, x22])+atol) and
          (np.min([y21, y22])-atol <= y_int <= np.max([y21, y22])+atol)):
    return False

  return True


def neighbor_list_to_adj(neighbor_list):
  n = len(neighbor_list)
  adj = np.zeros((n, n))
  for i, nbrs in enumerate(neighbor_list):
    adj[i, nbrs] = 1.
  return adj

