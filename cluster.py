#!/usr/bin/python

import numpy as np
from scipy.sparse import coo_matrix

import phenograph

def get_knn(distance, k=20):
    topk = np.argsort(distance)[:, 1:k+1]
    values, row, col, kstart = [], [], [], -(k+1)
    for cell in np.arange(topk.shape[0]):
        knn_c = topk[cell,:]
        row.append(cell*np.ones((k,)))
        col.append(knn_c)
        values.append(distance[cell, knn_c])
    indices = (np.hstack(row), np.hstack(col))
    return coo_matrix((np.hstack(values), indices),shape=distance.shape)


def run_phenograph(distance, k=20, outdir='', prefix=''):
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
    communities, graph, Q = phenograph.cluster(knn)

    if outdir is not None and len(outdir)>0:
        fileprefix = '{}/{}'.format(outdir, prefix)
        clusterfile = fileprefix + '.pg.txt'
        np.savetxt(clusterfile, communities)

        logfile = fileprefix + '.pg.info.txt'
        with open(logfile, 'w') as f:
            f.write('k:{}\nQ:{}'.format(k, Q))

    return communities, graph, Q

