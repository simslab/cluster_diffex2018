#!/usr/bin/python

import numpy as np
from scipy.sparse import coo_matrix

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
    if k > distance.shape[0]:
        raise ValueError(f'k {k} < number of cells {distance.shape[0]}')
    topk = np.argsort(distance)[:, 1:k+1]
    values, row, col, kstart = [], [], [], -(k+1)
    for cell in np.arange(topk.shape[0]):
        knn_c = topk[cell,:]
        row.append(cell*np.ones((k,)))
        col.append(knn_c)
        values.append(distance[cell, knn_c])
    indices = (np.hstack(row), np.hstack(col))
    return coo_matrix((np.hstack(values), indices),shape=distance.shape)


def run_phenograph(distance, k=20, outdir='', prefix='', **kwargs):
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
    knn = get_knn(distance, k)
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

