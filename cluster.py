#!/usr/bin/python

import numpy as np
from scipy.sparse import csr_matrix

import phenograph

def get_knn(distance, k=20):
    """
    Parameters
    ----------
    distance : np.array
        cell x cell distances
    k : int, optional
        number of nearest neighbors [30]

    Returns
    -------
    knn_coo : coo_matrix
        cell x cell coo matrix where each row contains `k` nonzero entries
        corresponding to the cell's k nearest neighbors.  Entries are set to
        with the distance between the two corresponding cells
    """
    topk = np.argsort(distance)[:, 1:k+1]

    #make into csr matrix
    indptr = [0]
    indices = []
    data = []
    for cell in np.arange(topk.shape[0]):
        indices.extend(topk[cell, :])
        data.extend(distance[cell, topk[cell, :]])
        indptr.append(len(indices))
    csr = csr_matrix((data,indices, indptr))
    return csr.tocoo()


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

