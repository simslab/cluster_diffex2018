#!/usr/bin/python
import numpy as np

def _import_plotlibs(for_save=True):
    """Import plotlibs properly for either save or jupyter notebook.
    Really annoying to have to do this as a method, but one or the other always
    freaks out if imports aren't conditioned on context.
    """
    if for_save:
        import matplotlib as mpl
        mpl.use('agg')
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        warnings.filterwarnings("ignore", category=UserWarning,
            module="matplotlib")
    else:
        import matplotlib as mpl
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('ticks')
    return mpl, plt, sns


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




