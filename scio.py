#!/usr/bin/python

import numpy as np
import pandas as pd


def load_gene_by_cell_matrix(matrix_file):
    """
    Read a gene expression matrix with no header and rows of the form:
    ENSG    GENE_SYMBOL    CELL_0    CELL_1    ...
    Return a gene x cell dataframe and a (ensg, gene symbol) dataframe
    """
    df = pd.read_csv(matrix_file, delim_whitespace=True, header=None)
    genes = df[[0,1]].copy()
    genes.columns = ['ens', 'gene']
    del df[0], df[1]
    df.columns = np.arange(df.columns.size)

    return df, genes


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


