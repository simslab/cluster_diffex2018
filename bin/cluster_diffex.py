import os
import gc
import argparse
import numpy as np
import pandas as pd

from clusterdiffex.util import load_txt
from clusterdiffex.distance import select_markers, get_distance
from clusterdiffex.cluster import run_phenograph
from clusterdiffex.visualize import run_umap, run_dca, plot_clusters
from clusterdiffex.diffex import binomial_test_cluster_vs_rest, \
    write_diffex_by_cluster, diffex_heatmap


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count-matrix', required=True,
            help='Input data. Should be a whitespace-delimited gene by'
            ' cell UMI count matrix with two leading columns of gene'
            ' attributes: ENSEMBL_ID and GENE_NAME. The file may not have'
            ' a header.')
    parser.add_argument('-o', '--outdir', required=True,
            help='The output directory.')
    parser.add_argument('-p', '--prefix', default='',
            help='A prefix to prepend to output filename (outfiles have the'
            ' form {OUTDIR}/{PREFIX}.markers.txt for example).')
    parser.add_argument('-n', '--norm', default='none',
            choices=['none', 'cp10k', 'log2cp10k'],
            help='Normalization to use.')
    parser.add_argument('-r', '--dim-redux', default='marker',
            choices=['none', 'marker', 'pca'],
            help='Dimensionality reduction to use as input to distance'
            ' calculation.  Currently only `marker` implemented.')
    parser.add_argument('-d', '--distance', default='spearman',
            choices=['spearman', 'euclidean', 'pearson', 'cosine', 'jaccard',
                     'hamming', 'energy', 'earthmover', 'braycurtis',
                     'canberra'],
            help='The distance metric to use.')
    parser.add_argument('-k', default=20, type=int,
            help='Number of nearest neighbors to use for clustering.')

    # marker selection/loading parameters
    parser.add_argument('-mf', '--marker-file', default='', type=str,
            help='Use the given marker file rather than determining highly '
            'variable genes from `count-matrix` (for marker based column '
            'selection).')
    parser.add_argument('--nstd', default=6.0, type=float,
            help='Only used when `dim-redux`=`marker` and `marker_file` not '
            'given. Sets adaptive threshold for marker selection at `nstd` '
            'standard devations above the mean dropout score. The threshold '
            'used is min(adaptive_threshold, absolute_theshold).')
    parser.add_argument('--absolute-threshold', default=0.15, type=float,
            help='Only used when `dim-redux`=`marker` and `marker_file` not '
            'given. Sets absolute threshold for marker selection. The threshold '
            'used is min(adaptive_threshold, absolute_theshold).')
    parser.add_argument('--window-size', default=25, type=int,
            help='Only used when `dim-redux`=`marker` and `marker_file` not '
            'given. Sets size of rolling window used to approximate expected '
            'fraction of cells expressing given the mean expression.')

    # visualization
    # parser.add_argument('--tsne', action='store_true', default=False)
    # parser.add_argument('--no-tsne', dest='tsne', action='store_false')

    parser.add_argument('--dmap', dest='dmap', action='store_true', default=True)
    parser.add_argument('--no-dmap', dest='dmap', action='store_false')

    return parser


def _parseargs_post(args):
    # set norm to none if values same w/ and w/out norm for distance
    norm_free_metrics = ['spearman', 'cosine']
    if args.distance in norm_free_metrics and args.norm != 'none':
        msg = 'Distance metric {} invariant to normalization.'
        msg += ' Setting norm to `none` (given {}).'
        print(msg.format(args.distance, args.norm))
        args.norm = 'none'

    binerized_metrics = ['jaccard', 'hamming']
    if args.distance in binerized_metrics and args.dim_redux == 'pca':
        msg = '{} requires binerized data, and cannot be applied to {} '
        msg += 'transformed data.'
        raise ValueError(msg.format(args.distance))

    if args.distance in binerized_metrics and args.norm != 'none':
        msg = 'Distance metric {} will be run on a binerized matrix.'
        msg += ' Setting norm to `none` (given {}).'
        print(msg.format(args.distance, args.norm))
        args.norm = 'none'

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    return args


def _get_distance_label(metric):
    """get label for distance metric"""
    if metric == 'spearman':
        return 'dcorrSP'
    elif metric == 'pearson':
        return 'dcorrPR'
    elif metric in ['earthmover', 'wasserstein']:
        return 'earthmover'
    elif metric == 'euclidean':
        return 'euc'
    elif metric == 'cosine':
        return 'cos'
    else:
        return metric


if __name__=='__main__':
    parser = _parser()
    args = parser.parse_args()
    args = _parseargs_post(args)

    # load the count matrix
    print('Loading UMI count matrix')
    counts, genes = load_txt(args.count_matrix)
    counts = pd.DataFrame(counts.T.A)
    genes.columns = ['ens', 'gene']
    nonzero = counts.sum(axis=1) > 0
    counts = counts.loc[nonzero]
    genes = genes.loc[nonzero]

    # start the running prefix
    running_prefix = [args.prefix]

    # normalize if needed
    if args.norm in ['cp10k', 'log2cp10k']:
        norm = counts / counts.sum(axis=0) * 10000
        running_prefix.append(args.norm)
        if args.norm == 'log2cp10k':
            norm = np.log2(norm + 1)
    else:
        norm = counts

    # select markers or reduce dimensionality
    if args.dim_redux == 'marker':
        if len(args.marker_file):
            # load markers from file
            loaded_markers = pd.read_csv(args.marker_file, header=None,
                    delim_whitespace=True,)
            # select markers that are present in the count matrix
            markers = genes.loc[genes.ens.isin(loaded_markers[0])]
            # get indices of markers in count matrix
            marker_ix = np.where(genes.ens.isin(markers.ens).values)[0]
            # check that we didn't mess up
            check_names_sorted = genes.iloc[marker_ix].ens.sort_values()
            loaded_names_sorted = markers.ens.sort_values()
            assert(all(check_names_sorted.values==loaded_names_sorted.values))

            msg = 'Found {} marker gene names in {}. {} matching genes found '
            msg += 'in count matrix.'
            print(msg.format(len(markers.ens.unique()), args.marker_file,
                             len(marker_ix)))
        else:
            # pick our own markers using the dropout curve
            marker_ix = select_markers(counts.values, outdir=args.outdir,
                prefix=args.prefix, gene_names=genes,
                window=args.window_size, nstd=args.nstd,
                t=args.absolute_threshold)
        running_prefix.append('markers')
        redux = norm.iloc[marker_ix]
    elif args.dim_redux in ['none', 'pca']:
        print('{} not yet implemented'.format(args.dim_redux))
        raise(ValueError())

    # get distance
    metric_label = _get_distance_label(args.distance)
    running_prefix.append(metric_label)
    distance = get_distance(redux, metric=args.distance, outdir=args.outdir,
                            prefix='.'.join(running_prefix))

    # visualize
    umap = run_umap(distance, prefix='.'.join(running_prefix),
            outdir=args.outdir)
    if args.dmap:
        try:
            dca = run_dca(distance, prefix='.'.join(running_prefix),
                    outdir=args.outdir)
        except RuntimeError as e:
            print('DCA error: {}'.format(e))
            dca = None

    # if args.tsne:
        # print('tsne not yet implemented')

    # cluster
    communities, graph, Q = run_phenograph(distance, k=args.k,
            prefix='.'.join(running_prefix), outdir=args.outdir)
    nclusters = len(np.unique(communities[communities > -1]))
    print('{} clusters identified by Phenograph'.format(nclusters))
    # visualize communities
    plot_clusters(communities, umap, outdir=args.outdir,
            prefix='.'.join(running_prefix + ['umap']))
    if args.dmap and dca is not None:
        print(args.dmap)
        plot_clusters(communities, dca, outdir=args.outdir,
                prefix='.'.join(running_prefix + ['dca']),
                label_name='Diffusion Component')

    # differential expression
    up, down, cluster_info = binomial_test_cluster_vs_rest(counts, genes,
            communities, '.'.join(running_prefix), for_gsea=True, verbose=True)
    write_diffex_by_cluster(up, down, args.outdir, cluster_info)
    diffex_heatmap(counts, genes, communities, up, 10, args.outdir,
            '.'.join(running_prefix))
