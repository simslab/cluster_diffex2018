import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from clusterdiffex.util import load_txt, load_loom
from clusterdiffex.distance import select_markers, get_distance, \
    select_markers_static_bins_unscaled
from clusterdiffex.cluster import run_phenograph
from clusterdiffex.visualize import run_umap, run_dca, run_tsne, plot_clusters
from clusterdiffex.diffex import binomial_test_cluster_vs_rest, \
    write_diffex_by_cluster, diffex_heatmap


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count-matrix', required=True,
            help='Input data. Should be a whitespace-delimited gene by'
            ' cell UMI count matrix with two leading columns of gene'
            ' attributes: ENSEMBL_ID and GENE_NAME. The file may not have'
            ' a header. Can also be a loom file.')
    parser.add_argument('-o', '--outdir', required=True,
            help='The output directory.')
    parser.add_argument('-p', '--prefix', default='',
            help='A prefix to prepend to output filename (outfiles have the'
            ' form {OUTDIR}/{PREFIX}.markers.txt for example).')
    parser.add_argument('-n', '--norm', default='none',
            choices=['none', 'cp10k', 'log2cp10k'],
            help='Normalization to use.')
    parser.add_argument('-d', '--distance', default='spearman',
            choices=['spearman', 'euclidean', 'pearson', 'cosine', 'jaccard',
                     'hamming', 'energy', 'earthmover', 'braycurtis',
                     'canberra'],
            help='The distance metric to use.')
    parser.add_argument('-k', default=20, type=int,
            help='Number of nearest neighbors to use for clustering.')

    parser.add_argument('--save-distance', action='store_true', default=False,
            help='Save the distance matrix (we don\'t do this by default'
            ' because the file is big but easy to recalculate).')

    # marker selection/loading parameters
    parser.add_argument('--absolute-threshold', default=0.15, type=float,
            help= 'Sets absolute threshold for marker selection. In the'
            ' default case with scaled dropout scores, the final threshold'
            ' used is min(adaptive_threshold, absolute_theshold). Only'
            ' the absolute threshold is used if option --unscaled score'
            ' is given. Ignored if a --marker-file is given.')
    parser.add_argument('--nstd', default=6.0, type=float,
            help= 'Sets adaptive threshold for marker selection at `nstd`'
            ' standard devations above the mean dropout score. The threshold'
            ' used is min(adaptive_threshold, absolute_theshold). Ignored with'
            ' option `--unscaled-score` or if a `--marker-file` is given.')
    parser.add_argument('--window-size', default=25, type=int,
            help='Sets size of rolling window used to approximate expected'
            ' fraction of cells expressing given the mean expression. Ignored'
            ' with option `--unscaled-score` or if a `--marker-file` is given.')
    parser.add_argument('-mf', '--marker-file', default='', type=str,
            help='Use the given marker file rather than determining highly '
            'variable genes from `count-matrix` (for marker based column '
            'selection).')
    parser.add_argument('--unscaled-score', action='store_true', default=False,
            help='Use an unscaled score with fixed bins for marker selection'
            ' rather than the default scaled score with fixed bin.')

    # visualization
    parser.add_argument('--tsne', action='store_true', default=False)
    parser.add_argument('--no-tsne', dest='tsne', action='store_false')

    parser.add_argument('--dmap', dest='dmap', action='store_true',
            default=False, help='Compute and and plot diffusion map.')
    parser.add_argument('--no-dmap', dest='dmap', action='store_false',
            help='Do not compute or plot diffusion map.')

    # cluster params
    parser.add_argument('-mcs', '--min-cluster-size', default=10, type=int,
            help='Minimum cluster size for phenograph.')

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

    timestr = time.strftime("%Y%m%d-%H%M%S")

    # load the count matrix
    print('Loading UMI count matrix')
    cellinfo = None
    if args.count_matrix.endswith('.loom'):
        counts, genes, cellinfo = load_loom(args.count_matrix)
    else:
        counts, genes = load_txt(args.count_matrix)
    counts = pd.DataFrame(counts.T.A)
    genes.columns = ['ens', 'gene']
    nonzero = counts.sum(axis=1) > 0
    counts = counts.loc[nonzero]
    genes = genes.loc[nonzero]

    # start the running prefix
    running_prefix = [args.prefix]

    # save the arguments
    arg_file = '{}/{}.cluster_diffex_args.json'.format(args.outdir, args.prefix)
    args.starttime = timestr
    print('Writing args to {}'.format(arg_file))
    with open(arg_file, 'w') as f:
        json.dump(args.__dict__, f,  indent=2)

    # normalize if needed
    if args.norm in ['cp10k', 'log2cp10k']:
        norm = counts / counts.sum(axis=0) * 10000
        running_prefix.append(args.norm)
        if args.norm == 'log2cp10k':
            norm = np.log2(norm + 1)
    else:
        norm = counts

    # select markers or reduce dimensionality
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
    elif args.unscaled_score:
        marker_ix = select_markers_static_bins_unscaled(counts.values,
                outdir=args.outdir, prefix=args.prefix, gene_names=genes,
                t=args.absolute_threshold)
    else:
        # pick our own markers using the dropout curve
        marker_ix = select_markers(counts.values, outdir=args.outdir,
                prefix=args.prefix, gene_names=genes,
                window=args.window_size, nstd=args.nstd,
                t=args.absolute_threshold)
    running_prefix.append('markers')
    redux = norm.iloc[marker_ix]

    # get distance
    metric_label = _get_distance_label(args.distance)
    running_prefix.append(metric_label)
    distance = get_distance(redux, metric=args.distance,
            outdir=args.outdir if args.save_distance else '',
            prefix='.'.join(running_prefix))

    # cluster
    communities, graph, Q = run_phenograph(distance, k=args.k,
            prefix='.'.join(running_prefix), outdir=args.outdir,
            min_cluster_size=args.min_cluster_size)
    nclusters = len(np.unique(communities[communities > -1]))
    print('{} clusters identified by Phenograph'.format(nclusters))

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

    if args.tsne:
        tsne = run_tsne(distance, prefix='.'.join(running_prefix),
                outdir=args.outdir)

    # visualize communities
    plot_clusters(communities, umap, outdir=args.outdir,
            prefix='.'.join(running_prefix + ['umap']))
    if args.dmap and dca is not None:
        plot_clusters(communities, dca, outdir=args.outdir,
                prefix='.'.join(running_prefix + ['dca']),
                label_name='Diffusion Component')
    if args.tsne and tsne is not None:
        plot_clusters(communities, tsne, outdir=args.outdir,
                prefix='.'.join(running_prefix + ['tsne']),
                label_name='tSNE')

    # differential expression
    up, down, cluster_info = binomial_test_cluster_vs_rest(counts, genes,
            communities, '.'.join(running_prefix), for_gsea=True,
            verbose=True)
    diffex_outdir = '{}/{}.diffex_binom/'.format(args.outdir,
            '.'.join(running_prefix))
    if os.path.exists(diffex_outdir):
        diffex_outdir.replace('_binom', '_binom-' + timestr)

    write_diffex_by_cluster(up, down, diffex_outdir, cluster_info)
    diffex_heatmap(counts, genes, communities, up, 10, diffex_outdir,
            '.'.join(running_prefix))

    if cellinfo is not None:
        print('Writing cell info from loom')
        cellinfo_file = '{}/{}.cells.txt'.format(args.outdir,args.prefix)
        cellinfo.to_csv(cellinfo_file, sep='\t', index=False)



