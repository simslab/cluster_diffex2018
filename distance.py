#!/usr/bin/python

import numpy as np
from scipy.stats.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

from util import _import_plotlibs

"""
Functions for marker gene selection and computing cell-cell distances or
similarities
"""

def get_spearman(matrix, outdir='', prefix='', verbose=False):
    """ Computes the Spearman's correlation from an expression matrix

    Parameters
    ----------
    matrix : ndarray
        matrix of elements compare similarity on.  Might be a counts matrix, a
        counts matrix of selected genes, PC components, etc.
    outdir: str, default ''
    prefix: str, default ''
    verbose: bool (default False)

    Returns
    -------
    sp_matrix : matrix of cell-cell Spearman's correlation coefficients

    """
    if verbose:
        print('Computing Spearman correlation matrix...')
    sp_matrix = spearmanr(matrix)[0]
    if outdir is not None and len(outdir)>0:
        filename='{}/{}.corrSP.txt'.format(outdir, prefix.rstrip('.'))
        # write Spearman correlation matrix to file
        print('Writing correlation matrix...')
        np.savetxt(filename, sp_matrix, delimiter='\t')
    return sp_matrix


def get_pearson(matrix, outdir='', prefix='', verbose=False):
    """ Computes the Pearson's correlation from an expression matrix

    Parameters
    ----------
    matrix : ndarray
        matrix of elements compare similarity on.  Might be a counts matrix, a
        counts matrix of selected genes, PC components, etc.
    outdir: str, default ''
    prefix: str, default ''
    verbose: bool (default False)

    Returns
    -------
    sp_matrix : matrix of cell-cell Spearman's correlation coefficients

    """
    if verbose:
        print('Computing Pearson correlation matrix...')
    pr_matrix = np.corrcoef(matrix)
    if outdir is not None and len(outdir)>0:
        filename='{}/{}.corrPR.txt'.format(outdir, prefix.rstrip('.'))
        # write  correlation matrix to file
        print('Writing correlation matrix...')
        np.savetxt(filename, pr_matrix, delimiter='\t')
    return pr_matrix

def get_jaccard_distance(matrix, outdir, prefix, threshold=0):
    """Compute pairwise jaccard on binerized matrix

    Parameters
    ----------
    matrix : ndarray
        matrix of molecular counts or PC loadings, etc to binerize
    outdir: str
        output directory
    prefix :str
        name of sequencing data set (e.g. PJ015)
    threshold : float, default 0
        binerization threshold

    Results
    -------
    d_matrix : cell by cell distance matrix
    """
    bin_matrix = np.where(matrix > threshold, np.ones_like(matrix),
            np.zeros_like(matrix))
    return get_distance(bin_matrix, outdir, prefix, metric='jaccard')


def get_distance(matrix, outdir, prefix, metric='euclidean',
        alt_metric_label=''):
    """ get and write pairwise distance

    Parameters
    ----------
    matrix : ndarray
        matrix of molecular counts or PC loadings, etc
    outdir: str
        output directory
    prefix :str
        name of sequencing data set (e.g. PJ015)
    metric : str, default 'euclidean'
        valid metric input to scipy.spatial.distance.pdist
    alt_metric_label : str, default ''
        name to use in filenames for metric instead metric name.
        only used if len(`alt_metric_label`) > 0

    Results
    -------
    d_matrix : cell by cell distance matrix
    """
    print('Computing distance matrix...')
    d_matrix = squareform(pdist(matrix.T, metric=metric))
    # write Spearman correlation matrix to file
    print('Writing distance matrix...')
    metric_label = alt_metric_label if len(alt_metric_label) > 0 else metric
    outfile = '{0}/{1}.{2}.txt'.format(outdir, prefix, metric_label)
    np.savetxt(outfile, d_matrix, delimiter='\t')
    return d_matrix


def select_markers(counts, window=25, nstd=6, t=0.15,
        outdir='', prefix='', gene_names=None):
    """
    Parameters
    ----------
    counts : ndarray
        gene x cell count matrix
    window : int, (default 25)
        size of window centered at each gene
    nstd : float, (default 6)
        number of standard deviations from the mean to set an adaptive
        dropout threshold.  To force use of a hard threshold, set to
        something really high.
    t : float (default 0.15)
        maximum threshold for designation as a dropout gene
    verbose : bool (default True)
        verbose output
    outdir: str, default ''
    prefix: str, default ''
    genes : pandas dataframe, optional
        ordered gene names and any other info.  must have integer indices.


    Returns
    -------
    ix_passing : ndarray
        indices of passing genes
    """
    print("Found {} genes in {} cells...".format(counts.shape[0], counts.shape[1]))
    print("Calculating dropout scores...")
    dropout, means, scores = _dropout_scores(counts, window)

    adaptive_threshold = nstd*np.std(scores) + np.mean(scores)
    threshold = min(adaptive_threshold, t)
    if threshold == adaptive_threshold:
        msg = 'Using adaptive threshold {adaptive_threshold}'
        msg += ' over absolute threshold {t}'
    else:
        msg = 'Using absolute threshold {t}'
        msg += ' over adaptive threshold {adaptive_threshold}'
    print(msg.format(adaptive_threshold=adaptive_threshold, t=t))
    ix_passing = np.arange(counts.shape[0])[scores > threshold].astype(int)

    n_markers = len(ix_passing)
    print('Found {} markers from dropout analysis'.format(n_markers))

    # write things to file
    if outdir is not None and len(outdir) > 0:
        # record parameters and adaptive threshold
        print('Writing threshold info...')
        thresholdfile = '{}/{}.dropout_threshold.txt'.format(outdir, prefix)
        with open(thresholdfile, 'w') as f:
            msg = 'nstdev: {}\nadaptive:{}\nt: {}\n'.format(nstd,
                    adaptive_threshold, t)
            f.write(msg)

        # save marker indexes
        ixfile = '{}/{}.marker_ix.txt'.format(outdir, prefix)
        np.savetxt(ixfile, ix_passing.astype(int), fmt='%i')

        # save marker gene names if gene_names given
        if gene_names is not None:
            markerfile = '{}/{}.markers.txt'.format(outdir, prefix)
            passing_names = gene_names.iloc[ix_passing]
            passing_names.to_csv(markerfile, sep='\t', header=None, index=None)

        # plot the dropout curve
        print('Plotting dropout curve')
        # annoying import trickery to avoid exceptions due to matplotlib's
        # backend in different contexts
        mpl, plt, sns = _import_plotlibs()
        from matplotlib.backends.backend_pdf import PdfPages
        pdffile = '{}/{}.dropout_curve.pdf'.format(outdir, prefix)
        with PdfPages(pdffile) as pdf:
            plt.plot(means,dropout,'ko',
                     means[ix_passing], dropout[ix_passing],'go',
                     markersize=4)
            plt.ylim([-0.05,1.05])
            plt.xlabel('log10(Mean Normalized Counts)')
            plt.ylabel('Fraction of Cells')
            pdf.savefig()
            plt.close()

    return ix_passing


def _dropout_scores(counts, window=25):
    """Score genes based on their deviation from the dropout curve.

    Genes are ordered by mean normalized expression, and a gene's expected
    observation rate is estimated as the maximum observation rate in a
    window of size `window` centered on the gene. A gene's score is its
    scaled devation from its expected rate:
    (expected_rate - observed_rate) / sqrt(expected_rate).

    Parameters
    ----------
    counts : ndarray
        gene x cell count matrix
    window : int, (default 25)
        size of window centered at each gene

    Returns
    -------
    dropout : counts.shape[0] x 1 ndarray
        fraction of cells in which gene is observed
    means : counts.shape[0] x 1 ndarray
        log10( normalized  mean expression), where each cell's
        expression is normalized to sum to 1
    scores : counts.shape[0] x 1 ndarray
        dropout scores
    """
    # normalize and sort by  means
    normed = counts/np.sum(counts, axis=0)
    means = np.log10(np.mean(normed, axis=1))
    gene_order = np.argsort(means)
    sorted_means = means[gene_order]

    dropout = np.count_nonzero(counts, axis=1) / counts.shape[1]
    sorted_dropout = dropout[gene_order]
    # get max fraction of cells expressing in rolling window around sorted gene
    rolling_max_truncated = np.max(_rolling_window(sorted_dropout, window),
            axis=1)
    # pad edges of window
    size_diff = sorted_dropout.shape[0] - rolling_max_truncated.shape[0]
    pad_amt = [int(np.floor(size_diff/2)), int(np.ceil(size_diff/2))]
    rolling_max = np.pad(rolling_max_truncated, pad_amt, 'edge')
    # get score
    score = (rolling_max - sorted_dropout) / np.sqrt(rolling_max)
    # reorder scores to match original matrix ordering
    score_reorder = score[np.argsort(gene_order)]

    return dropout, means, score_reorder


def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
