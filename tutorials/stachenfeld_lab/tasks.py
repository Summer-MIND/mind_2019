"""Functions for generating task graphs, plotting positions, and labels."""
import numpy as np
from scipy.linalg import block_diag
import scipy.spatial.distance as distance
import scipy.spatial as spatial
import scipy.stats as stats
from sklearn.manifold import MDS


def line(n):
  """Get the adjacency matrix for a linear track.

  Args:
    n: number of states

  Returns:
    adjmat: adjacency matrix
    xy: xy coordinates of each state for plotting
    labels: all zero (shape (n,))
  """
  adjmat_upper_tri = np.concatenate(
      [np.concatenate([np.zeros((n-1, 1)), np.eye(n-1)], axis=1),
       np.zeros((1, n))],
      axis=0)

  xy = np.concatenate([np.linspace(0, 1, n).reshape(-1, 1), np.zeros((n, 1))],
                      axis=1)

  return adjmat_upper_tri + adjmat_upper_tri.T, xy, np.zeros(len(adjmat_upper_tri))


def ring(n):
  """Get the adjacency matrix for a linear track.

  Args:
    n: number of states

  Returns:
    adjmat: adjacency matrix
    xy: xy coordinates of each state for plotting
    labels: all zero (shape (n,))
  """
  adjmat_upper_tri = np.concatenate(
      [np.concatenate([np.zeros((n-1, 1)), np.eye(n-1)], axis=1),
       np.zeros((1, n))],
      axis=0)
  adjmat_upper_tri[0, n-1] = 1

  angles = np.linspace(0, 2 * np.pi, n+1)[:n]
  x, y = (np.cos(angles), np.sin(angles))
  xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

  return adjmat_upper_tri + adjmat_upper_tri.T, xy, np.zeros(len(adjmat_upper_tri))


def rectangle_mesh(dims):
  """Rectangular mesh adjacency matrix and positions.

  Args:
    dims: tuple containing dimensions of graph

  Returns:
    adj: adjacency matrix
    pos: positions of each node in Rn where n = len(dims)
    labels: all zero (shape (n,))
  """
  mesh = np.meshgrid(*[np.arange(d) for d in dims])
  xy = np.stack(mesh).reshape(len(dims), -1).T

  dist_mat = distance.squareform(distance.pdist(xy, "euclidean"))
  adj = np.array(dist_mat <= 1., dtype=np.float32)
  adj = adj - np.diag(np.diag(adj))
  return adj, xy, np.zeros(len(adj))


def tower_of_hanoi():
  """Get 27x27 tower of hanoi adjacency matrix, xy coordinates, and labels.

  Returns:
    adjmat: adjacency matrix
    xy: xy coordinates of each state for plotting. Obtained by arranging nodes
      within a cluster into evenly spaced circles, and then arranging those
      clusters evenly around a circle.
    labels: (n_state) array containing cluster index of each state
  """
  # to make general: arbitrary meshgrid
  clu_list = [i.reshape(-1, 1) for i in
              np.meshgrid(np.arange(3), np.arange(3), np.arange(3))]
  clu = np.concatenate([clu_list[i] for i in [1, 0, 2]], axis=1)

  # TODO(stachenfeld): generalize to allow for arbitrary depth
  # to make general: make 2**i for i in range(n_levels-1, -1)
  # or  2**i * .5 for i in range(n_levels-2, -2, -1) or
  y_scale = np.sqrt(3)/2
  ys = np.dot(clu == 2, np.array([4, 2, 1])) * y_scale
  xs = (np.dot(clu == 1, np.array([4, 2, 1])) +
        np.dot(clu == 2, np.array([2, 1, .5])))
  node2xy = np.concatenate([i.reshape(-1, 1) for i in [xs, ys]], axis=1)

  adjmat = np.array(distance.pdist(node2xy, metric="euclidean") < 1.5,
                    dtype=np.double)

  adjmat = distance.squareform(adjmat)
  xy = np.concatenate([i.reshape(-1, 1) for i in [xs, ys]], axis=1)

  return adjmat, xy, clu[:, 0]


def clique(n):
  """Return adjacency matrix for a clique.

  Args:
    n: number of states

  Returns:
    adjmat: adjacency matrix
    xy: xy coordinates of each state for plotting
    labels: all zero (shape (n,))
  """

  adj = np.ones(n) - np.eye(n)
  angles = np.linspace(0, 2 * np.pi, n+1)[:n]
  x, y = (np.cos(angles), np.sin(angles))
  xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
  return adj, xy, np.zeros(len(adj))


def clique_ring(n_cluster=3, n_in_cluster=5):
  """Get adjacency matrix for cluster domain used by Schapiro et al 2013.

  Args:
    n_cluster: number of clusters, connected in a ring.
    n_in_cluster: number of nodes in each cluster. Each node is connected to all
      other nodes in cluster, except the edge connecting the two nodes with
      outgoing edges is severed.

  Returns:
    adjmat: adjacency matrix
    xy: xy coordinates of each state for plotting. Obtained by arranging nodes
      within a cluster into evenly spaced circles, and then arranging those
      clusters evenly around a circle.
    labels: (n_state) array containing cluster index of each state
  """
  n_state = n_cluster * n_in_cluster

  clq, _, _ = clique(n_in_cluster)
  clq[0, n_in_cluster-1] = 0
  clq[n_in_cluster-1, 0] = 0

  adj = clq
  for i in range(n_cluster-1):
    adj = block_diag(adj, clq)

  for i in range(n_cluster):
    i_curr = np.mod(i * n_in_cluster-1, n_state)
    i_next = np.mod(i * n_in_cluster, n_state)
    adj[i_curr, i_next] = 1
    adj[i_next, i_curr] = 1

  # get xy
  clu_ind = np.repeat(np.arange(0, n_cluster).reshape(-1, 1),
                      n_in_cluster, axis=0).reshape(-1)
  ang_clu = clu_ind * 1.0 / n_cluster * 2 * np.pi
  x_clu = np.cos(ang_clu).reshape(-1, 1) * 2
  y_clu = np.sin(ang_clu).reshape(-1, 1) * 2

  offset = np.pi - ang_clu - np.pi/n_in_cluster  # turn clusters toward center
  ang_in_clu = np.linspace(0, 2*np.pi, n_in_cluster+1)[:n_in_cluster]
  ang_in_clus = np.stack([ang_in_clu]*n_cluster).reshape(-1)

  ang_in_clus = ang_in_clus - offset
  x_in_clu = np.cos(ang_in_clus).reshape(-1, 1)
  y_in_clu = np.sin(ang_in_clus).reshape(-1, 1)

  # get cluster labels
  labels = np.concatenate([np.ones(n_in_cluster) * i for i in range(n_cluster)])

  return adj, np.concatenate([x_clu+x_in_clu, y_clu+y_in_clu], axis=1), labels


def tree(depth, n_branch, directed=True):
  """Get adjacency matrix for tree maze.

  Args:
    depth: depth of maze
    n_branch: number of branches
    directed: if True, return adjacency matrix for directed tree (from root ->
      leaves). Else return undirected matrix (directed_adj+directed_adj.T). Note
      from leaves -> root tree is directed_adj.T

  Returns:
    adj: nxn adjacency matrix
    pos: nx2 array of x, y positions of nodes for plotting (y=maze depth,
      x=position at depth)
    root_leaves: n length vector conveying if node is a root (-1) (or source),
      leaf (+1) (or sink), or neither (0)
      or neither (0)
  """
  n = np.sum(n_branch**np.arange(0, depth+1))

  adj = np.zeros((n, n))
  y = np.concatenate([np.zeros(n_branch**i)+i for i in range(depth+1)])
  x = np.concatenate([np.arange(n_branch**i) - np.mean(np.arange(n_branch**i))
                      for i in range(depth+1)])
  pos = np.stack([x, y]).T
  pos[:, 0] = pos[:, 0] * 2**(pos[:, 1].max() - pos[:, 1])
  pos[:, 1] = -pos[:, 1]

  i = 0
  j = 1
  while j < n:
    adj[i, j:j+n_branch] = 1.
    j = j + n_branch
    i += 1

  adj = adj if directed else adj+adj.T
  root_leaves = (np.array(pos[:, 1] == np.max(pos[:, 1]), dtype=np.float32) -
                 np.array(pos[:, 1] == np.min(pos[:, 1]), dtype=np.float32))
  return adj, pos, root_leaves


def stochastic_block(n_clu, n_in_cluster, clu_p):
  """Create a stochastic block matrix.

  Args:
    n_clu: scalar, number of clusters
    n_in_cluster: iterable of len n_clu, number of nodes in each cluster
    clu_p: n_clu x n_clu array, wi_clu_p[i, j] relays the probability of a
      connection from node k to l given node k is in cluster i and l is in j

  Returns:
    adjmat: adjacency matrix
    xy: xy coordinates of each state for plotting (found with multidimensional
      scaling, 2 components)
    labels: (n_state) array containing cluster membership of each state
  """
  # fill in adjacency matrix with within-cluster and between-cluster adj_ij's
  clu_p = np.array(clu_p)
  np.testing.assert_equal(clu_p, clu_p.T)

  adj = []
  for i in range(n_clu):
    adj.append([])
    for j in range(n_clu):
      ni, nj = (n_in_cluster[i], n_in_cluster[j])
      if i < j:
        adj_ij = np.zeros((ni, nj))
      else:
        adj_ij = np.random.binomial(1, clu_p[i, j], size=(ni, nj))
        if i == j:
          adj_ij = np.triu(adj_ij, 1)  # lower triangle to 0
      adj[i].append(adj_ij)
    adj[i] = np.concatenate(adj[i], axis=1)
  adj = np.concatenate(adj, axis=0)
  diag_adj = np.diag(adj)
  if len(diag_adj.shape) == 1:
    diag_adj = np.diag(adj)
  adj -= np.diag(adj)

  embedding = MDS(n_components=2)
  xy = embedding.fit_transform(adj)
  labels = np.concatenate([np.array([i] * n_in_cluster[i]) for i in range(n_clu)])

  return adj+adj.T, xy, labels


def four_rooms(dims, doorway=1.):
    """
    Args:
      dims: [dimx, dimy] dimensions of rectangle
      doorway: size of doorway

    Returns:
      adjmat: adjacency matrix
      xy: xy coordinates of each state for plotting
      labels: empty []
    """
    half_x, half_y = (dims[0]*.5, dims[1]*.5)
    quarter_x, quarter_y = (dims[0]*.25, dims[1]*.25)
    threequarter_x, threequarter_y = (dims[0]*.75, dims[1]*.75)

    adj, xy, _ = rectangle_mesh(dims)
    room = np.array([xy[:,0] < half_x, xy[:,1] < half_y], dtype=np.float32).T
    mask = np.array(distance.squareform(distance.pdist(room, "euclidean")) == 0, dtype=np.float32)
    labels = np.sum(room * np.array([[1, 2]]), 1)

    doorsx = [quarter_x, threequarter_x, half_x, half_x]
    doorsy = [half_y, half_y, quarter_y, threequarter_y]
    doors = np.array([doorsx, doorsy]).T
    inds = []
    for d in doors:
        dist_to_door = np.sum(np.abs(xy - d[None, :]), 1)
        ind = np.where(dist_to_door == np.min(dist_to_door))[0]
        if len(ind) > 1: ind = ind[0]
        mask[ind, :] = 1
        mask[:, ind] = 1

    adj = adj * mask
    return adj, xy, labels
