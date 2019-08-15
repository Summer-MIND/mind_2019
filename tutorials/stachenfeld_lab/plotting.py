"""Plotting functions."""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sklearn.manifold
import seaborn as sns
import os
import util
import warnings


def plot_graph(adj, ax=None, xy=None, show_ticks=False,
               node_size=300, node_color=None, cmap=None, vmin=None, vmax=None,
               show_node_labels=False, skip_edges=False, atol=1e-7):
  """Plot a supplied graph."""
  graph = nx.from_numpy_array(adj)
  if xy is None:
    xy = nx.spring_layout(graph)

  if node_color is None:
    node_color = np.zeros((len(adj), 3))

  # if node_color variation is likely numerical error, replace with mean
  if node_color.max() - node_color.min() < atol:
      node_color = node_color * 0 + node_color.mean()

  if (vmin is None) and (vmax is None):
    vmax = np.max(np.abs(node_color))
    if np.min(node_color) < 0:
      vmin = -1 * vmax
      if cmap is None:
        cmap = "RdBu"
    else:
      vmin = np.min(node_color)

  if ax is None:
    _, ax = plt.subplots(1)

  nodes = nx.draw_networkx_nodes(graph, pos=xy, node_size=node_size,
                                 node_color=node_color, cmap=cmap, vmin=vmin,
                                 vmax=vmax, ax=ax)

  nodes.set_edgecolor("k")
  # Suppressing networkx deprecation error (...you probably shouldn't really do this)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if not skip_edges:
      nx.draw_networkx_edges(graph, pos=xy, ax=ax)
  if show_node_labels:
    nx.draw_networkx_labels(graph, pos=xy)
  if not show_ticks:
    ax.set_xticks([])
    ax.set_yticks([])


def plot_many_features(adj, xy, features, nrows=3, ncols=3, figsize=(12, 10), axes=None,
                       node_size=300, norm_per_feature=False, cmap=None):
  """Plot first nrows*ncols features over graph."""
  if axes is None:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
  else: fig = None

  # Get colormap range.
  if not norm_per_feature:
    vmax = np.max([features.max(), -features.min()])
    if features.min() < 0:
      vmin = -vmax
      cmap = "RdBu" if cmap is None else cmap
    else:
      vmin = features.min()
      cmap = sns.cubehelix_palette(light=1, as_cmap=True) if cmap is None else cmap
  else:
      vmin = None
      vmax = None
      if features.min() < 0:
        cmap = "RdBu" if cmap is None else cmap
      else:
        cmap = sns.cubehelix_palette(light=1, as_cmap=True) if cmap is None else cmap

  k = 0
  for i in range(nrows):
    for j in range(ncols):
      if k < features.shape[1]:
        plot_graph(
            adj, ax=axes[i][j], xy=xy, show_ticks=False, node_size=node_size, node_color=features[:, k],
            cmap=cmap, vmin=vmin, vmax=vmax)
      axes[i][j].axis("equal")
      axes[i][j].set_xticks([])
      axes[i][j].set_yticks([])
      k += 1
  return fig, axes


def plot_adj(adj, include_ticks=False, ax=None):
  "Plot heatmap of adjacency matrix."""
  if ax is None:
    ax = plt.gca()
  sns.heatmap(adj, ax=ax, cbar=False, vmin=0, vmax=1)
  if not include_ticks:
    ax.set_xticks([])
    ax.set_yticks([])


