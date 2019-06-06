#!/usr/bin/python
import warnings
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd


def sample_molecules(counts, n):
    """Sample n molecules from a vector of molecular counts

    Parameters
    ----------
    counts: ndarray
        ngene x 1 count vector
    n: int
        number of molecules to sample

    Returns
    -------
    samples: ndarray
        ngene x 1 count vector of sampled molecules
    """
    # sample molecular indices
    selected = np.random.permutation(np.sum(counts))[:n]
    # bin molecules into genes
    gene_bins = np.append(np.array([0]), np.cumsum(counts))
    downsampled = np.histogram(selected, gene_bins)[0]
    return downsampled


def load_txt(filename,  ngene_cols=2, verbose=True,):
    """Load data from a whitespace delimited txt file

    From https://github.com/simslab/scHPF/blob/master/schpf/preprocessing.py

    Parameters
    ----------
    filename : str
        file to load.  Expected to be a gene x cell whitespace-delimited file
        without a header where the first `ngene_cols` are gene identifiers,
        names or other metadata.
    ngene_cols : int, optional (default: 2)
        The number of columns that contain row attributes (ie gene id/names)
    verbose : bool, optional (default: True)
        print progress messages

    Returns
    -------
    coo : coo_matrix
        cell x gene sparse count matrix
    genes : pd.DataFrame
        ngenes x ngene_cols array of gene names/attributes
    """
    assert( ngene_cols > 0 )
    gene_cols = list(range(ngene_cols))

    if filename.endswith('.gz') or filename.endswith('.bz2'):
        msg = '.....'
        msg+= 'WARNING: Input file {} is compressed. '.format(filename)
        msg+= 'It may be faster to manually decompress before loading.'
        print(msg)

        df = pd.read_csv(filename, header=None, memory_map=True,
                delim_whitespace=True)

        genes = df[gene_cols]
        dense = df.drop(columns=gene_cols).values.T
        nz = np.nonzero(dense)
        coo = coo_matrix((dense[nz], nz), shape=dense.shape, dtype=np.int32)
    else:
        genes, rows, cols, values = [], [], [], []

        # load row by row to conserve memory + actually often faster
        with open(filename) as f:
            # for each gene/row
            for g, l in enumerate(f):
                llist = l.split()
                genes.append(llist[:ngene_cols])
                r, c, val = [], [], []

                # for each cell/column
                for cell,v in enumerate(llist[ngene_cols:]):
                    if v != '0':
                        r.append(int(cell))
                        c.append(int(g))
                        val.append(int(v))

                rows.extend(r)
                cols.extend(c)
                values.extend(val)

                if verbose and ((g+1)%10000 == 0) and (g!=0):
                    print('\tloaded {} genes for {} cells'.format(
                        g+1, cell+1))

        ncells, ngenes = len(llist[ngene_cols:]), g+1
        coo = coo_matrix((np.array(values), (np.array(rows),np.array(cols))),
                shape=(ncells,ngenes), dtype=np.int32)
        genes = pd.DataFrame(genes)

    return coo, genes


def load_loom(filename):
    """Load data from a loom file

    From github.com/simslab/scHPF

    Parameters
    ----------
    filename: str
        file to load

    Returns
    -------
    coo : coo_matrix
        cell x gene sparse count matrix
    genes : Dataframe
        Dataframe of gene attributes.  Attributes are ordered so
        Accession and Gene are the first columns, if those attributs are
        present
    cells : Dataframe
        Dataframe of cell attributes
    """
    import loompy
    # load the loom file
    with loompy.connect(filename) as ds:
        loom_genes = pd.DataFrame(dict(ds.ra.items()))
        loom_cells = pd.DataFrame(dict(ds.ca.items()))
        loom_coo = ds.sparse().T

    # order gene attributes so Accession and Gene are the first two columns,
    # if they are present
    first_cols = []
    for colname in ['Accession', 'Gene']:
        if colname in loom_genes.columns:
            first_cols.append(colname)
    rest_cols = loom_genes.columns.difference(first_cols).tolist()
    loom_genes = loom_genes[first_cols + rest_cols]

    return loom_coo,loom_genes,loom_cells


def write_gene_by_cell_matrix(genes, matrix, outfile):
    """
    Takes a gene x cell dataframe and a (ensg, gene symbol) dataframe
    Writes a gene expression matrix with rows of the form:
    """
    pd.concat([genes, matrix], axis=1).to_csv(outfile, header=False,
            index=False, sep='\t')


def load_cluster_file(cluster_file, name='cluster', index=[]):
    """
    Load file with a single column of integers representing clusters
    """
    clusters = pd.Series(np.loadtxt(cluster_file, dtype=np.int16), name=name)
    if len(index):
        clusters.index = index
    return clusters




