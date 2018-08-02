#! /usr/bin/python

import numpy as np
import umap
import dmaps

from util import _import_plotlibs

def run_umap(distance, outdir, prefix):
    """ Compute and plot a 2D umap projection from a distance matrix.

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
        cell x 2 array of umap embedding coordinates

    """
    # get appropriate libraries in a context-specific manner
    _, plt, _ = _import_plotlibs()
    from matplotlib.backends.backend_pdf import PdfPages

    # umap
    print('Running umap...')
    umap_model = umap.UMAP(metric='precomputed', )
    embedding = umap_model.fit_transform(distance)

    outfile_coords = '{}/{}.umap.txt'.format(outdir, prefix)
    outfile_pdf = '{}/{}.umap.pdf'.format(outdir, prefix)

    # plot
    print('Plot umap output...')
    np.savetxt(outfile_coords, embedding, delimiter='\t')

    with PdfPages(umap_PDF) as pdf:
        plt.plot(embedding[:,0],embedding[:,1],'ko',markersize=4)
        plt.xlabel('UMAP Axis 1')
        plt.ylabel('UMAP Axis 2')
        ax = plt.gca()
        ax.set_aspect('equal')
        pdf.savefig()
        plt.close()
    return embedding


def run_dca(distance, outdir, prefix):
    """ Compute and plot a 2D umap projection from a distance matrix.

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
    # get appropriate libraries in a context-specific manner
    _, plt, _ = _import_plotlibs()
    from matplotlib.backends.backend_pdf import PdfPages

    # get diffusion commpontnets
    print('Running DCA...')
    dmap = dmaps.DiffusionMap(matrix)
    dmap.set_kernel_bandwidth(3)
    dmap.compute(3)
    dmap_eig = dmap.get_eigenvectors()
    embedding = np.array([dmap_eig[:,1]/dmap_eig[:,0],
                          dmap_eig[:,2]/dmap_eig[:,0]]).T

    outfile_coords = '{}/{}.dca.txt'.format(outdir, prefix)
    outfile_pdf = '{}/{}.dca.pdf'.format(outdir, prefix)

    # plot
    print('Plot dca output...')
    np.savetxt(outfile_coords, embedding, delimiter='\t')

    with PdfPages(umap_PDF) as pdf:
        plt.plot(embedding[:,0],embedding[:,1],'ko',markersize=4)
        plt.xlabel('Diffusion Component 1')
        plt.ylabel('Diffusion Component 2')
        ax = plt.gca()
        ax.set_aspect('equal')
        pdf.savefig()
        plt.close()
    return embedding


def plot_clusters(clusters, coordinates, outdir, prefix, label_name='UMAP'):
    # get appropriate libraries in a context-specific manner
    mpl, plt, _ = _import_plotlibs()
    from matplotlib.backends.backend_pdf import PdfPages

    outfile = '{}/{}.pg.pdf'.format(outdir, prefix)

    colors = ['red','green','blue','magenta','brown','cyan','black','orange',
              'grey','darkgreen','yellow','tan','seagreen','fuchsia','gold',
              'olive']
    N = len(set(communities))
    if N > len(colors):
        colors = [name for name,hex in mpl.colors.cnames.items()]
        colors.reverse()

    with PdfPages(outfile) as pdf:
        fig,ax = plt.subplots()
        for c in range(N):
            my_coords = coordiates[clusters==c, :]
            ax.scatter(my_coords[:,0], my_coords[:,1], color=colors[c], s=10,
                    lw=0, edgecolor='none', label=str(c))
        plt.xlabel('{} 1'.format(label_name))
        plt.ylabel('{} 2'.format(label_name))
        ax.set_aspect('equal')
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
        pdf.savefig()
        plt.close()
