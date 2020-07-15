#!/usr/bin/env python

import os
import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmread, mmwrite
from clusterdiffex.distance import spearmanr
import umap

import phenograph

def get_knn(distance, k=20):
    """ Get a cell x cell k-nearest neigbors adjacency matrix

    Parameters
    ----------
    distance : ndarray
        cell by cell distance matrix
    k : int, optional (Default: 20)
        Number of neighbors to include in graph

    Returns
    -------
    knn : coo_matrix
        sparse knn adjacency matrix where each row has k nonzero entries whose
        values are the distance between the cell for the row and its k nearest
        neighbors
    """
    topk = np.argsort(distance)[:, 1:k+1]
    values, row, col, kstart = [], [], [], -(k+1)
    for cell in np.arange(topk.shape[0]):
        knn_c = topk[cell,:]
        row.append(cell*np.ones((k,)))
        col.append(knn_c)
        values.append(distance[cell, knn_c])
    indices = (np.hstack(row), np.hstack(col))
    return coo_matrix((np.hstack(values), indices),shape=distance.shape)


def get_approx_knn(X, k=20, metric=spearmanr, metric_kw={}, angular=True,
        random_state=0):
    """
    Parameters
    -----------
    X: ndarray
        cell x gene
    k: int

    """
    from sklearn.utils import check_random_state
    cols, vals, _ = umap.umap_.nearest_neighbors(X, k, metric, metric_kw,
            angular, check_random_state(random_state))
    cols_flat = np.hstack(cols)
    vals_flat = np.hstack(vals)
    rows = np.ravel(np.column_stack( [np.arange(cols.shape[0])]*k ))
    assert len(cols_flat) == len(vals_flat)
    assert len(cols_flat) == len(rows)
    return coo_matrix((vals_flat, (rows, cols_flat)), shape=(X.shape[0],
        X.shape[0]))


def run_phenograph_approx_knn(X, k=20, outdir='', prefix='', **kwargs):
    """
    Runs Phenograph on an expression- or PCA-based distance matrix.

    Parameters
    ----------
    X: ndarray
        cell x feature data matrix
    k: int (default 20)
        number of nearest neighbors to use
    outdir: str (default '')
    prefix: str (default '')
    label: str (default '')

    Returns
    -------
    communities
    graph
    Q : float

    """
    assert X.shape[0] > X.shape[1]
    fileprefix = '{}/{}'.format(outdir, prefix)
    knn_file = f'{fileprefix}.knn{k}_approx.mtx'
    if os.path.exists(knn_file):
        knn = mmread(knn_file)
    else:
        knn = get_approx_knn(X, k) #.tolil()
        mmwrite(knn_file, knn)

    print(83, knn.shape)
    communities, graph, Q = phenograph.cluster(knn, **kwargs)

    if outdir is not None and len(outdir)>0:
        fileprefix = '{}/{}'.format(outdir, prefix)
        clusterfile = fileprefix + '.pg.txt'
        np.savetxt(clusterfile, communities, fmt='%i')

        logfile = fileprefix + '.pg.info.txt'
        with open(logfile, 'w') as f:
            f.write('k:{}\nQ:{}'.format(k, Q))

    return communities, graph, Q


def run_phenograph(distance, k=20, outdir='', prefix='', approx_knn=False,
        **kwargs):
    """
    Runs Phenograph on an expression- or PCA-based distance matrix.

    Parameters
    ----------
    distance: ndarray
        cell x cell distance matrix
    k: int (default 20)
        number of nearest neighbors to use
    outdir: str (default '')
    prefix: str (default '')
    label: str (default '')

    Returns
    -------
    communities
    graph
    Q : float

    """
    knn = get_knn(distance, k).tolil()
    communities, graph, Q = phenograph.cluster(knn, **kwargs)

    if outdir is not None and len(outdir)>0:
        fileprefix = '{}/{}'.format(outdir, prefix)
        clusterfile = fileprefix + '.pg.txt'
        np.savetxt(clusterfile, communities, fmt='%i')

        logfile = fileprefix + '.pg.info.txt'
        with open(logfile, 'w') as f:
            f.write('k:{}\nQ:{}'.format(k, Q))

    return communities, graph, Q

def cluster_mask_generator(clusters):
    """ Get all cluster masks

    Parameters
    ----------
    clusters: ndarray
        cell x 1 vector of cluster labels

    Returns
    -------
    cluster_generator : generator
        generate (label, mask) pairs for each cluster
    """
    for c in np.sort(np.unique(clusters)):
        cluster_mask = clusters==c
        yield c, cluster_mask


def paired_cluster_mask_generator(clusters):
    """ Get masks for all pairs of (different) clusters

    Parameters
    ----------
    clusters: ndarray
        cell x 1 vector of cluster labels

    Returns
    -------
    cluster_generator : generator
        generate [(label_0, cluster_mask_0), (label_1, cluster_mask_1)] pairs
        for all 2 member combinations of clusters (no replacement)
    """
    for c0 in np.sort(np.unique(clusters)):
        for c1 in np.sort(np.unique(clusters)):
            if c0 < c1:
                yield [(c0,clusters==c0), (c1,clusters==c1)]

