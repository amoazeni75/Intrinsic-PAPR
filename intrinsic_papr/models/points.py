"""Growing the point cloud: choosing where to add points and how to combine neighbours."""

import numpy as np
import torch
from scipy.spatial import KDTree

def add_points_knn(
    coords,
    influ_scores,
    add_num,
    k,
    sample_type,
    sample_k=10,
    point_features=None,
    acc_coord_grad_norm=None,
    hybrid_weight=0.5,
):
    """
    Add points to the point cloud by kNN
    """
    kdtree = KDTree(coords)
    N = coords.shape[0]

    # Step 1: Determine where to add points
    if N == 0:
        return None, 0, None, None
    if N <= add_num:
        inds = np.random.choice(N, add_num, replace=True)
        query_coords = coords[inds, :]
    else:
        if sample_type == "acc-coord-grad-norm-max":
            inds = np.argsort(acc_coord_grad_norm)[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-norm-max-hybrid-top-knn-std":
            inds_a = np.argsort(acc_coord_grad_norm)
            ranks_a = np.zeros_like(inds_a)
            ranks_a[inds_a] = np.arange(len(inds_a))

            assert k >= 2
            nns_dists, nns_inds = kdtree.query(coords, k=sample_k + 1)
            nns_dists = nns_dists[:, 1:]
            inds_b = np.argsort(nns_dists.std(axis=-1))
            ranks_b = np.zeros_like(inds_b)
            ranks_b[inds_b] = np.arange(len(inds_b))

            ranks = hybrid_weight * ranks_a + (1 - hybrid_weight) * ranks_b
            inds = np.argsort(ranks)[-add_num:]
            query_coords = coords[inds, :]
        else:
            raise NotImplementedError(
                "point sample type [{:s}] is not supported".format(sample_type)
            )

    # Step 2: Add points by kNN, combining each neighbourhood with random weights
    new_features = None
    nns_dists, nns_inds = kdtree.query(query_coords, k=k + 1)
    nns_dists = nns_dists.astype(np.float32)
    nns_dists = nns_dists[:, 1:]
    nns_inds = nns_inds[:, 1:]
    rnd_w = np.random.uniform(0, 1, (query_coords.shape[0], k)).astype(np.float32)
    rnd_w /= rnd_w.sum(axis=-1, keepdims=True)
    new_coords = (coords[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
    new_influ_scores = (influ_scores[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
    if point_features is not None:
        new_features = (point_features[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(
            axis=-2
        )
    return new_coords, len(new_coords), new_influ_scores, new_features


def normalize_vector(x, eps=0.0):
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)
