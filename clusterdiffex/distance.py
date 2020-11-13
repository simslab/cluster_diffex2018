#!/usr/bin/python

"""
Functions for marker gene selection and computing cell-cell distances or
similarities
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import numba

try:
    from scipy.stats import energy_distance, wasserstein_distance
except ImportError:
    msg = 'Warning: could not import energy_distance or wasserstein_distance. '
    msg+= 'To use energy or earthmover distance, upgrade scipy.'
    print(msg)

from clusterdiffex.visualize import _import_plotlibs


def get_distance(matrix, outdir='', prefix='', metric='spearman'):
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
    d_matrix : ndarray
        cell by cell distance matrix
    """
    print('Computing {} distance matrix...'.format(metric))

    if metric=='spearman':
        distance = 1 - stats.spearmanr(matrix)[0]
    elif metric == 'pearson':
        distance = 1 - np.corrcoef(matrix.T)
    elif metric in ['jaccard', 'hamming']:
        binerized = np.where(matrix > 0, np.ones_like(matrix),
                np.zeros_like(matrix))
        distance = squareform(pdist(binerized.T, metric=metric))
    elif metric == 'energy':
        distance = squareform(pdist(matrix.T, metric=energy_distance))
    elif metric in ['earthmover', 'wasserstein']:
        distance = squareform(pdist(matrix.T, metric=wasserstein_distance))
    else:
        distance = squareform(pdist(matrix.T, metric=metric))

    # write Spearman correlation matrix to file
    if outdir is not None and len(outdir):
        print('Writing distance matrix...')
        outfile = '{0}/{1}.txt'.format(outdir, prefix)
        np.savetxt(outfile, distance, delimiter='\t')
    return distance


def select_markers(counts, window=25, nstd=6, t=0.15,
        outdir='', prefix='', gene_names=None):
    """ Select marker with rolling window and scaling

    Procedure used in Levitin et al. 2019 and Szabo, Levitin et al.  2019.
    For selection method used in Yuan et al. 2018 and Mizrak et al. 2019, see
    `select_markers_static_bins`

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
    outdir: str, optional (Default: '')
        If given, directory to save markers and plots to
    prefix: str, optional (Default: '')
        If given, prefix for save filenames
    gene_names : pandas dataframe, optional
        ordered gene names and any other info. must have integer indices.  If
        given, Used to write a file with marker gene names in addition to
        indices.

    Returns
    -------
    ix_passing : ndarray
        indices of selected genes


    TODO: refactor w/ select_markers_static_bins_unscaled to avoid repeated code
    """
    print("Found {} genes (with a nonzero count) in {} cells...".format(
        counts.shape[0], counts.shape[1]))
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
    ix_passing = np.where(scores > threshold)[0]

    n_markers = len(ix_passing)
    print('Found {} markers from dropout analysis'.format(n_markers))

    # write things to file
    if outdir is not None and len(outdir) > 0:
        # record parameters and adaptive threshold
        my_prefix = prefix.rstrip('.') + '.' if len(prefix) else ''
        print('Writing threshold info...')
        thresholdfile = '{}/{}dropout_threshold.txt'.format(outdir, my_prefix)
        with open(thresholdfile, 'w') as f:
            msg = 'nstdev: {}\nadaptive:{}\nt: {}\n'.format(nstd,
                    adaptive_threshold, t)
            f.write(msg)

        # save marker indexes
        print('Saving marker gene indexes...')
        ixfile = '{}/{}marker_ix.txt'.format(outdir, my_prefix)
        np.savetxt(ixfile, ix_passing, fmt='%i')

        # save marker gene names if gene_names given
        if gene_names is not None:
            print('Saving marker gene names...')
            markerfile = '{}/{}markers.txt'.format(outdir, my_prefix)
            passing_names = gene_names.iloc[ix_passing]
            passing_names.to_csv(markerfile, sep='\t', header=None, index=None)

        # plot the dropout curve
        print('Plotting dropout curve')
        # annoying import trickery to avoid exceptions due to matplotlib's
        # backend in different contexts
        mpl, plt, sns = _import_plotlibs()
        from matplotlib.backends.backend_pdf import PdfPages
        pdffile = '{}/{}dropout_curve.pdf'.format(outdir, my_prefix)
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


def select_markers_static_bins_unscaled(counts, t=0.2, outdir='', prefix='',
        gene_names=None):
    """Select markers with fixed bins and no scaling

    Procedure used in Yuan et al. 2018 and Mizrak et al. 2019. For marker
    selection algorithm used in Levitin et al. 2019 and Szabo, Levitin, et al.
    2019, see `select_markers`

    Parameters
    ----------
    counts : ndarray
        gene x cell count matrix
    t : float, optional (Default: 0.15)
        maximum threshold for designation as a dropout gene
    verbose : bool (default True)
        verbose output
    outdir: str, optional (Default: '')
        If given, directory to save markers and plots to
    prefix: str, optional (Default: '')
        If given, prefix for save filenames
    gene_names : pandas dataframe, optional
        ordered gene names and any other info. must have integer indices.  If
        given, Used to write a file with marker gene names in addition to
        indices.

    Returns
    -------
    ix_passing : ndarray
        indices of selected genes

    TODO: refactor w/ select_markers to avoid repeated code
    """
    print("Found {} genes (with a nonzero count) in {} cells...".format(
        counts.shape[0], counts.shape[1]))
    print("Calculating dropout scores...")
    dropout, means, scores = _dropout_scores_static_bins_unscaled(counts)
    ix_passing = np.where(scores > t)[0]

    n_markers = len(ix_passing)
    print('Found {} markers from dropout analysis'.format(n_markers))

    # write things to file
    if outdir is not None and len(outdir) > 0:
        # record parameters and adaptive threshold
        my_prefix = prefix.rstrip('.') + '.' if len(prefix) else ''
        print('Writing threshold info...')
        thresholdfile = '{}/{}dropout_threshold.txt'.format(outdir, my_prefix)
        with open(thresholdfile, 'w') as f:
            msg = 't: {}\n'.format(t)
            f.write(msg)

        # save marker indexes
        print('Saving marker gene indexes...')
        ixfile = '{}/{}marker_ix.txt'.format(outdir, my_prefix)
        np.savetxt(ixfile, ix_passing, fmt='%i')

        # save marker gene names if gene_names given
        if gene_names is not None:
            print('Saving marker gene names...')
            markerfile = '{}/{}markers.txt'.format(outdir, my_prefix)
            passing_names = gene_names.iloc[ix_passing]
            passing_names.to_csv(markerfile, sep='\t', header=None, index=None)

        # plot the dropout curve
        print('Plotting dropout curve')
        # annoying import trickery to avoid exceptions due to matplotlib's
        # backend in different contexts
        mpl, plt, sns = _import_plotlibs()
        from matplotlib.backends.backend_pdf import PdfPages
        pdffile = '{}/{}dropout_curve.pdf'.format(outdir, my_prefix)
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
    pass


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


def _dropout_scores_static_bins_unscaled(counts, bin_size=50):
    """Score genes based on their deviation from the dropout curve (older
    method).

    Genes are ordered by mean normalized expression, and sorted into bins of
    `bin_size` genes. A gene's expected observation rate is estimated as the
    maximum observation rate in its bin. The gene's score is its deviation
    from its expected rate: expected_rate - observed_rate

    Parameters
    ----------
    counts : ndarray
        gene x cell count matrix
    bin_size : int, optional (Default: 50)
        size of window centered at each gene

    Returns
    -------
    dropout : counts.shape[0] x 1 ndarray
        fraction of cells in which gene is observed
    means : counts.shape[0] x 1 ndarray
        log10( normalized  mean expression), where each cell's
        expression is normalized to sum to 1
    scores : counts.shape[0] x 1 ndarray
        unscaled dropout scores estimated from static bins
    """
    # normalize and sort by  means
    normed = counts/np.sum(counts, axis=0)
    means = np.log10(np.mean(normed, axis=1))
    gene_order = np.argsort(means)
    sorted_means = means[gene_order]

    dropout = np.count_nonzero(counts, axis=1) / counts.shape[1]
    sorted_dropout = dropout[gene_order]

    nbins = int(np.ceil(len(means)/bin_size))
    binmax = np.zeros_like(means)
    for i in range(nbins):
        start = i*bin_size
        end = min((i+1)*bin_size, len(means))
        binmax[start:end] = np.max(sorted_dropout[start:end])
    score = binmax - sorted_dropout
    score_reorder = score[np.argsort(gene_order)]

    return dropout, means, score_reorder


def _rolling_window(a, window):
    """Internal method"""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# taken from pynndescent
@numba.njit()
def rankdata(a, method="average"):
    arr = np.ravel(np.asarray(a))
    if method == "ordinal":
        sorter = arr.argsort(kind="mergesort")
    else:
        sorter = arr.argsort(kind="quicksort")

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size)

    if method == "ordinal":
        return (inv + 1).astype(np.float64)

    arr = arr[sorter]
    obs = np.ones(arr.size, np.bool_)
    obs[1:] = arr[1:] != arr[:-1]
    dense = obs.cumsum()[inv]

    if method == "dense":
        return dense.astype(np.float64)

    # cumulative counts of each unique value
    nonzero = np.nonzero(obs)[0]
    count = np.concatenate((nonzero, np.array([len(obs)], nonzero.dtype)))

    if method == "max":
        return count[dense].astype(np.float64)

    if method == "min":
        return (count[dense - 1] + 1).astype(np.float64)

    # average method
    return 0.5 * (count[dense] + count[dense - 1] + 1)


# from pynndescent, except returns a distance
@numba.njit(fastmath=True)
def spearmanr(x, y):
    a = np.column_stack((x, y))

    n_vars = a.shape[1]

    for i in range(n_vars):
        a[:, i] = rankdata(a[:, i])
    rs = np.corrcoef(a, rowvar=0)

    return 1 - rs[1, 0]
