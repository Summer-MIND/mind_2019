"""The messiest test file in the world.

Runs 'smoke test' to make sure everything runs.
"""
import numpy as np
import tasks
import representations
import matplotlib.pyplot as plt
import plotting
import util
import seaborn as sns

def test_everything_runs():
    """Check everything runs."""
    discount = .9
    sigma = 1.
    n = 32  # line, ring params
    dims = [18, 22]  # rectangle, four rooms params
    n_clique = 5  # clique params
    # clique ring params
    n_cluster_clqrng = 3
    n_in_cluster_clqrng = 5
    # tree params
    depth = 3
    n_branch = 2
    # SBM params
    n_clu_sbm = 3
    n_in_clu_sbm = [10, 20, 25]
    p_between_cluster_sbm = .7*np.eye(3) + .2  # .9 w/i, .2 b/t

    outputs = [
            tasks.line(n),
            tasks.ring(n),
            tasks.rectangle_mesh(dims),
            tasks.four_rooms(dims),
            tasks.clique(n_clique),
            tasks.tower_of_hanoi(),
            tasks.clique_ring(n_cluster_clqrng, n_in_cluster_clqrng),
            tasks.tree(depth, n_branch, directed=False),
            tasks.stochastic_block(n_clu_sbm, n_in_clu_sbm, p_between_cluster_sbm),
    ]
    for adj, xy, labels in outputs:
        fig, ax = plt.subplots(1)

        # test reps
        randomwalk_transmat = util.l1_normalize_rows(adj)
        rep_onehot, _ = representations.state_onehot(randomwalk_transmat)
        rep_succ, _ = representations.successor_rep(randomwalk_transmat, discount)
        rep_Leig_unnorm, _ = representations.laplacian_eig(adj, "unnorm")
        rep_Leig_norm, _ = representations.laplacian_eig(adj, "norm")
        rep_Leig_rw, _ = representations.laplacian_eig(adj, "rw")
        rep_euc_gaussian, _ = representations.euclidean_gaussian(xy, sigma, None, 100)
        reps = [rep_onehot, rep_succ, rep_Leig_unnorm, rep_Leig_norm, rep_Leig_rw, rep_euc_gaussian]
        for r in reps:
          assert len(r.shape) == 2
          assert r.shape[0] == len(adj)

        node_color = labels if list(labels) else None
        node_color = rep_euc_gaussian[:, 0]
        plotting.plot_graph(adj, xy=xy, ax=ax, node_color=node_color, node_size=100)

    plt.show()


def main():
    test_everything_runs()
    print("Passed all tests!")

main()
