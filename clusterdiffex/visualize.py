#! /usr/bin/python

import warnings
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE


def _import_plotlibs(for_save=True):
    """Import plotlibs properly for either save or jupyter notebook.
    Really annoying to have to do this as a method, but one or the other always
    freaks out if imports aren't conditioned on context.

    Parameters
    ----------
    for_save : bool, optional (Default: True)
        is this in a script (True) or a notebook (False)

    Returns
    -------
    mpl : matplotlib
    plt : matplotlib.pyplot
    sns : seaborn
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


def get_cluster_cmap(N, mpl):
    if N < 11:
        colors = mpl.cm.tab10.colors
    elif N < 21 :
        colors = mpl.cm.tab20.colors
    else:
        colors = [name for name,hex in mpl.colors.cnames.items()]
        colors.reverse()
    return colors[:N]


def run_umap(data, outdir='', prefix='', metric='precomputed'):
    """ Compute and plot a 2D umap projection from a distance matrix.

    Parameters
    ----------
    data : ndarray
        cell x cell distance matrix or cell x feature matrix
    outdir : str, optional (Default: '')
        output directory for saving coordinates and plot
    prefix : str, optional (Default: '')
        prefix for file

    Returns
    -------
    embedding : ndarray
        cell x 2 array of umap embedding coordinates

    """

    # umap
    print('Running umap...')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        umap_model = UMAP(metric=metric)
        embedding = umap_model.fit_transform(data)

    if outdir is not None and len(outdir):
        _, plt, _ = _import_plotlibs()
        from matplotlib.backends.backend_pdf import PdfPages
        clean_prefix = prefix.rstrip('.') + '.' if len(prefix) else ''
        outfile_coords = '{}/{}umap.txt'.format(outdir, clean_prefix)
        outfile_pdf = '{}/{}umap.pdf'.format(outdir, clean_prefix)

        # plot
        print('Save UMAP coordinates...')
        np.savetxt(outfile_coords, embedding, delimiter='\t')

        print('Plot UMAP output...')

        with PdfPages(outfile_pdf) as pdf:
            plt.plot(embedding[:,0],embedding[:,1],'ko',markersize=4)
            plt.xlabel('UMAP Axis 1')
            plt.ylabel('UMAP Axis 2')
            ax = plt.gca()
            ax.set_aspect('equal')
            pdf.savefig()
            plt.close()
    return embedding


def run_dca(distance, outdir='', prefix=''):
    """ Compute and plot 2D diffusion embedding from a distance matrix.

    Parameters
    ----------
    distance : ndarray
        cell x cell distance matrix
    outdir : str
        output directory
    prefix : str
        prefix for file

    Returns
    -------
    embedding : ndarray
        cell x 2 array of first two diffusion components

    """
    # get diffusion commpontnets
    print('Running Diffusion Maps...')
    try:
        import dmaps
        dmap = dmaps.DiffusionMap(distance)
        dmap.set_kernel_bandwidth(3)
        dmap.compute(3)
        dmap_eig = dmap.get_eigenvectors()
        embedding = np.array([dmap_eig[:,1]/dmap_eig[:,0],
                            dmap_eig[:,2]/dmap_eig[:,0]]).T

        if outdir is not None and len(outdir):
            _, plt, _ = _import_plotlibs()
            from matplotlib.backends.backend_pdf import PdfPages
            clean_prefix = prefix.rstrip('.') + '.' if len(prefix) else ''
            outfile_coords = '{}/{}dca.txt'.format(outdir, clean_prefix)
            outfile_pdf = '{}/{}dca.pdf'.format(outdir, clean_prefix)

            # plot
            print('Save Diffusion Map coordinates...')
            np.savetxt(outfile_coords, embedding, delimiter='\t')

            print('Plot Diffusion Map output...')

            with PdfPages(outfile_pdf) as pdf:
                plt.plot(embedding[:,0],embedding[:,1],'ko',markersize=4)
                plt.xlabel('Diffusion Component 1')
                plt.ylabel('Diffusion Component 2')
                ax = plt.gca()
                ax.set_aspect('equal')
                pdf.savefig()
                plt.close()
        return embedding
    except ImportError:
        print('\nWARNING: could not compute or plot diffusion components'
                ' because dmaps is not installed. Install from https://'
                'github.com/hsidky/ dmaps. \n\nContinuing without...')


def run_tsne(distance, outdir='', prefix=''):
    """ Compute and plot 2D tSNE from a distance matrix.

    Parameters
    ----------
    distance : ndarray
        cell x cell distance matrix
    outdir : str
        output directory
    prefix : str
        prefix for file

    Returns
    -------
    embedding : ndarray
        cell x 2 array of first two diffusion components

    """
    print('Running tSNE...')
    tsne = TSNE(n_components=2, perplexity=20, learning_rate=1000,
            verbose=2, metric="precomputed", method="exact")
    embedding = tsne.fit_transform(distance)

    if outdir is not None and len(outdir):
        _, plt, _ = _import_plotlibs()
        from matplotlib.backends.backend_pdf import PdfPages
        clean_prefix = prefix.rstrip('.') + '.' if len(prefix) else ''
        outfile_coords = '{}/{}tsne.txt'.format(outdir, clean_prefix)
        outfile_pdf = '{}/{}tsne.pdf'.format(outdir, clean_prefix)

        # plot
        print('Save tSNE coordinates...')
        np.savetxt(outfile_coords, embedding, delimiter='\t')

        print('Plot tSNE output...')

        with PdfPages(outfile_pdf) as pdf:
            plt.plot(embedding[:,0],embedding[:,1],'ko',markersize=4)
            plt.xlabel('Diffusion Component 1')
            plt.ylabel('Diffusion Component 2')
            ax = plt.gca()
            ax.set_aspect('equal')
            pdf.savefig()
            plt.close()
    return embedding


def plot_clusters(clusters, coordinates, outdir, prefix, label_name='UMAP',
        outfile=None):
    """ Plot 2D embedding colored by cluster and save to file

    Note : currently only works for saving to file, not in notebooks

    Parameters
    ----------
    clusters : ndarray
        1D array of cluster labels
    coordinates : ndarray
        (ncells, 2) array of coordinates
    outdir  : str
        the output directory
    prefix : str
        prefix for file
    label_name: str, optional (Default: UMAP)
        Label name for axes
    outfile : str, optional (Default: None)
        If not none, ignore outdir  and prefix and use  for output
        instead
    """
    assert len(clusters) == len(coordinates)
    # get appropriate libraries in a context-specific manner
    mpl, plt, _ = _import_plotlibs()
    from matplotlib.backends.backend_pdf import PdfPages

    if outfile is None:
        outfile = '{}/{}.pg.pdf'.format(outdir, prefix)

    N = len(np.unique(clusters))
    colors = get_cluster_cmap(N, mpl)

    with PdfPages(outfile) as pdf:
        fig,ax = plt.subplots()
        for c in range(N):
            my_coords = coordinates[clusters==c, :]
            ax.scatter(my_coords[:,0], my_coords[:,1], color=colors[c], s=10,
                    lw=0, edgecolor='none', label=str(c))
        plt.xlabel('{} 1'.format(label_name))
        plt.ylabel('{} 2'.format(label_name))
        ax.set_aspect('equal')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        pdf.savefig()
        plt.close()
