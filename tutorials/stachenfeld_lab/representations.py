"""Functions for generating different representations."""
import numpy as np
from scipy.linalg import block_diag
import scipy.spatial.distance as distance
import scipy.spatial as spatial
import scipy.stats as stats


def state_onehot(transmat):
    """Returns identity matrix (corresponds to each state as onehot.
    Args:
      transmat: transition matrix (just used for dimensions)

    Returns:
      identity matrix of shape transmat.shape
      scores (all zeros, shape (n,))
    """
    return np.eye(len(transmat)), np.zeros(len(transmat))


def successor_rep(transmat, discount):
  """Compute successor representation matrix analytically, using the following.

  M = sum_(t=0)^infinity (discount^t T^t) = (I - discount*T)^(-1)  (eq. 7)

  Args:
    transmat: [n_state x n_state] transition matrix, where transmat[i, j] is
      equal to the probability of transitioning from state i from state j
    discount: scalar discount factor between [0 inclusive, 1 exclusive)

  Returns:
    srmat: successor representation matrix M, where srmat[i, j] is equal to the
      expected discounted number of visitations to state j starting from state
      i (eq. 3)
    scores (all zeros, shape (n,))
  """
  transmat = np.array(transmat, dtype=np.float32)
  n_state = transmat.shape[0]
  srmat = np.linalg.inv(np.eye(n_state) - discount * transmat)
  return srmat, np.zeros(len(srmat))


def laplacian(adj, norm="unnorm"):
  """Compute graph Laplacian.

  Args:
    adj: square adjacency matrix
    norm: what kind of normalized laplacian to apply ("unnorm": D-A,
      "norm": D^(-.5)(D-A)D^(-.5), "rw": D^(-1)(D-A))
      (default="unnorm")

  Returns:
    laplacian, square matrix
  """
  degree_vec = adj.sum(1)
  unnorm_laplacian = np.diag(degree_vec) - adj
  if norm == "unnorm":
    return unnorm_laplacian
  elif norm == "norm":
    norm = np.linalg.pinv(np.diag(np.sqrt(degree_vec)))
    return np.dot(np.dot(norm, unnorm_laplacian), norm)
  elif norm in ["rw", "randomwalk"]:
    norm = np.linalg.pinv(np.diag(degree_vec))
    return np.dot(norm, unnorm_laplacian)


def signed_amp(x):
  """Return sign(x) * amp(x), where amp is amplitude of complex number."""
  return np.sign(np.real(x)) * np.sqrt(np.real(x) ** 2 + np.imag(x) ** 2)


def eig(x, order="descend", sortby=signed_amp):
  """Computes eigenvectors and returns them in eigenvalue order.

  Args:
    x: square matrix to eigendecompose
    order: "descend" or "ascend" to specify in which order to sort eigenvalues
      (default="descend")
    sortby: function transforms a list of (possibly complex, possibly mixed
      sign) into real-valued scalars that can be sorted without ambiguity
      (default=signed_amp)

  Returns:
    evecs: array of eigenvectors
    evals: matrix with eigenvector columns
  """
  assert x.shape[0] == x.shape[1]
  n = x.shape[0]
  evals, evecs = np.linalg.eig(x)

  ind_order = range(n)
  ind_order = [x for _, x in sorted(zip(sortby(evals), ind_order))]
  if order == "descend":
    ind_order = ind_order[::-1]
  evals = evals[ind_order]
  evecs = evecs[:, ind_order]
  return evecs, evals


def laplacian_eig(adj, norm="norm", order="ascend"):
  """Return Laplacian eigenvectors.

  Args:
    adj: square adjacency matrix
    norm: what kind of normalized laplacian to apply ("unnorm": D-A,
      "norm": D^(-.5)(D-A)D^(-.5), "rw": D^(-1)(D-A))
      (default="norm")
    order: "descend" or "ascend" to specify in which order to sort eigenvalues
      (default="ascend")

  Returns:
    eigenvectors array shape=(n, n), each column eig[:, i] is an eigenvector.
    eigenvalues array shape=(n,) sorted in ascending or descending order (as
      specified) according to amplitude.
  """
  lap = laplacian(adj, norm)
  evecs, evals = eig(lap, order=order)
  return evecs, evals


def euclidean_gaussian(xy, sigma, centers=None, ncenters=None):
  """Return position in terms of Euclidean Gaussian RBFs.

  Args:
    xy: nstate x 2 array of xy positions
    sigma: sigma for Gaussian
    centers: locations of the Gaussian centers (if None, draw random or
      use xy)
    ncenters: if centers not supplied, use as number of Gaussian centers
      (if None, use xy as centers)

  Returns:
    State representations as Gaussian RBFs
    scores (all zeros, shape (n,))
  """
  if centers is None:
    if ncenters is None:
      centers = xy
    else:
      centers = np.random.rand(ncenters, 2)
      centers = centers * (xy.max(0) - xy.min(0)) + xy.min(0)

  rep = []
  for center in centers:
    rep.append(stats.multivariate_normal.pdf(xy, mean=center, cov=sigma))
  return np.array(rep).T, centers

