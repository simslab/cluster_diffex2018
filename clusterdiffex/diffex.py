#!/usr/bin/python

import os
import numpy as np
import pandas as pd
from scipy.stats import binom
from statsmodels.sandbox.stats.multicomp import multipletests

from clusterdiffex.cluster import cluster_mask_generator
from clusterdiffex.visualize import _import_plotlibs, get_cluster_cmap


class PopulationStats:
    """
    Class to hold population-level statistics (eg # of cells expressing each
    gene) for use in later analysis.
    """

    @staticmethod
    def create_from_expression(population_id, expression):
        """ Factory method to create a PopulationStats instance from an
            expression meatrix.

        Parameters
        ----------
        population_id: str
        expression : pandas dataframe
            gene x cell expression matrix

        Returns
        -------
        population_stats : PopulationStats instance
        """
        n_cells =     expression.shape[1]
        n_genes =     np.sum(expression, axis=1).astype(bool).sum()
        n_cells_exp = np.sum(expression.astype(bool), axis=1)
        n_mol =        np.sum(expression, axis=1)
        med_mol =      expression.median()

        return PopulationStats(population_id, n_cells=n_cells, n_genes=n_genes,
                n_cells_exp=n_cells_exp, n_mol=n_mol, med_mol=med_mol)


    def __init__(self, population_id, n_cells=0, n_genes=0, n_cells_exp=None,
            n_mol=None, med_mol=None):
        """
        Parameters
        ----------
        population_id : str
            id of the population
        n_cells : int
            number of cells in the population (total)
        n_genes : int
            number of genes expr. by cells in the population (total)
        n_cells_exp : pandas Dataframe or Series
            number of cells expr. each gene in population
        n_mol : pandas Dataframe or Series
            total number of molecules of each gene in population
        med_mol : pandas Dataframe or Series
            median number of molecules of each gene in population
        """
        self.id = population_id
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_cells_exp = n_cells_exp
        self.n_mol = n_mol
        self.med_mol = med_mol


    def sort(self, index):
        """Sort all pandas data structures according to index

        Parameters
        ----------
        index : pandas index or series
        """
        assert(index.sort_values().equals(self.n_cells_exp.index.sort_values()))
        self.n_cells_exp = self.n_cells_exp.loc[index]
        self.n_mol = self.n_mol.loc[index]
        if self.med_mol is not None:
            self.med_mol = self.med_mol.loc[index]


    def merge(self, other, new_id='', inplace=False):
        """ Merge with either a single or an iterable of other PopulationStats
            instances.

        Parameters
        ----------
        other: PopuluationStats instance or iterable of instances
        new_id : str
            id of the merged population stats. If the empty string,
            use self.id
        inplace : bool
            If true modify instance rather than returning new.  Note other's
            components are not modified if true, although the list itself is
            emptied.

        Returns
        -------
        population_stats : PopulationStats instance
            Only returned if inplace=False, otherwise no return

        Notes
        -----
        Merging invalidates med_mol (median transcripts), which is accordingly
        set to -1 in the merged PopulationStats instance.

        """
        if hasattr(other, '__iter__') and len(other)>0: # given a list
            to_merge = other.pop()
            if len(other) > 0:  # if more left in list, recurse
                to_merge = to_merge.merge(other, new_id)
            return self.merge(to_merge, new_id, inplace)
        else: # base case
            new_id = new_id if len(new_id) else self.id
            new_n_cells = self.n_cells + other.n_cells
            new_n_cells_exp = self.n_cells_exp.add(other.n_cells_exp,
                    fill_value=0)
            new_n_genes = len(new_n_cells_exp.index) if len(new_n_cells_exp) \
                    else self.n_genes
            new_n_mol = self.n_mol.add(other.n_mol, fill_value=0)
            # remove median transcripts (set to 1, cence)
            new_med_mol = None
            if not inplace:
                return PopulationStats(new_id, n_cells=new_n_cells,
                    n_genes=new_n_genes, n_cells_exp=new_n_cells_exp,
                    n_mol=new_n_mol, med_mol=new_med_mol)
            else:
                self.new_id = new_id
                self.n_cells = new_n_cells
                self.n_cells_exp = new_n_cells_exp
                self.n_genes = new_n_genes
                self.n_mol = new_n_mol
                self.med_mol = new_med_mol
                return None


def binomial_test(ingroup, outgroup, min_effectsize=2, FDR=0.01,
        min_proportion=0.2, correct_log_effect=True):
    """ Applies the binomial test to the proportions of cells expressing a
        gene in two populations as described in Shekhar et al., returning
        significantly enriched genes with respect to the ingroup

    Parameters
    ----------
    ingroup : PopulationStats instance
        cells to test
    outgroup : PopulationStats
        cells to test against
    min_effectsize: numeric
        Minimum magnitude of ratio of (Ng/N) / ((Mg)/M) for inclusion where
        Ng is the number of the cells in the ingroup expressing a gene, N is
        the number of cells in the ingroup, Mg is the number of cells in the
        outgroup expressing the gene, and M is the number cells in the
        outgroup. (default 2)
    FDR : float
        Max BH FDR for inclusion (default 0.01)
    min_proportion : float
        only comparisons with at least this proportion of cells expressing a
        gene in at least one population pass
    correct_log_effect : bool
        Handle cases where ingroup.n_cells_exp and/or outgroup.n_cells_exp are
        0.  Specifically, if 0 in both, set log2_effect to 0. If none in
        ingroup, set the proportion in the ingroup equal to 1/outgroup.n_cells,
        and the symmetrical operation if outgroup.n_cells_exp=0.

    Returns
    -------
    up : pandas Dataframe
        Upregulated genes in ingroup with respect to outgroup.
        Columns are ncluster (number expressin in ingroup), nrest (number
        expressing in outgroup), fdr, and log2_effect
    down : pandas Dataframe
        Downregulated genes in ingroup with respect to outgroup.  Columns
        are the same as in up.
    """
    assert(ingroup.n_cells_exp.index.equals(outgroup.n_cells_exp.index))

    # add regularizing pseudocounts for when population is in background
    in_exp1 = ingroup.n_cells_exp.copy()
    in_exp1[in_exp1 == 0] = 1
    out_exp1 = outgroup.n_cells_exp.copy()
    out_exp1[out_exp1 == 0] = 1

    # effect size
    with np.errstate(divide='ignore'):
        log2_ef = np.log2( (ingroup.n_cells_exp*outgroup.n_cells)
                / (out_exp1*(ingroup.n_cells)))
    # correct effect size for cases with 0s
    if correct_log_effect:
        no_in = ingroup.n_cells_exp == 0
        no_out = outgroup.n_cells_exp == 0
        log2_ef[no_in & ~no_out] = np.log2(1/outgroup.n_cells_exp[
            no_in & ~no_out])
        log2_ef[no_out & ~no_in] = np.log2(ingroup.n_cells_exp
                [no_out & ~no_in])
        log2_ef[no_in & no_out] = 0

    # calculate and correct pvals for up and down separately
    # make sure gammas are in same order as foreground
    gamma_up = out_exp1 / outgroup.n_cells
    p_up = binom.sf(ingroup.n_cells_exp, ingroup.n_cells, gamma_up) + binom.pmf(
            ingroup.n_cells_exp, ingroup.n_cells, gamma_up)
    _, q_up, _, _ = multipletests(p_up, method='fdr_bh')
    fdr_up = pd.Series(q_up, name='fdr', index=ingroup.n_cells_exp.index)

    gamma_down = in_exp1 / ingroup.n_cells
    p_down = binom.sf(outgroup.n_cells_exp, outgroup.n_cells, gamma_down
            ) + binom.pmf(outgroup.n_cells_exp, outgroup.n_cells, gamma_down)
    _, q_down, _, _ = multipletests(p_down, method='fdr_bh')
    fdr_down = pd.Series(q_down, name='fdr', index=outgroup.n_cells_exp.index)

    # put it togeterh to make a dataframe
    df_up = pd.concat([ingroup.n_cells_exp,outgroup.n_cells_exp,fdr_up,log2_ef],
        axis=1)
    df_up.columns = ['ncluster','nrest','fdr','log2_effect']
    df_down = pd.concat([ingroup.n_cells_exp,outgroup.n_cells_exp,fdr_down,
        log2_ef], axis=1)
    df_down.columns = ['ncluster','nrest','fdr','log2_effect']

    # filter
    sig_up = df_up.fdr < FDR
    sig_down = df_down.fdr < FDR
    if min_effectsize > 0:
        effect_up = df_up.log2_effect >= np.log2(min_effectsize)
        effect_down = df_down.log2_effect <= -np.log2(min_effectsize)
    else:
        # dummy mask to get all true
        effect_up = df_up.log2_effect == df_up.log2_effect
        effect_down = df_down.log2_effect == df_down.log2_effect
    if min_proportion > 0:
        prop_up = ingroup.n_cells_exp/ingroup.n_cells >= min_proportion
        prop_down = outgroup.n_cells_exp/outgroup.n_cells >= min_proportion
    else:
        prop_up, prop_down = effect_up, effect_down

    passing_up = sig_up & effect_up & (prop_up | prop_down)
    passing_down = sig_down & effect_down & (prop_up | prop_down)
    df_up_final = df_up.loc[passing_up].sort_values(
        by=['fdr', 'log2_effect'], ascending=[True,False])
    df_down_final = df_down.loc[passing_down].sort_values(
        by=['fdr', 'log2_effect'], ascending=[True,True])

    return df_up_final, df_down_final


def binomial_test_cluster_vs_rest(expression, genes, clusters,
        population_id, min_effectsize=2, FDR=0.01, min_proportion=0.2,
        aux=[], for_gsea=False, garbage_collect=False, verbose=False):
    """ Run binomial test on all samples in a cluster against all other
        samples in the cluster + stats for an optional aux variable.

    Parameters
    ----------
    expression : pandas Dataframe
    clusters : ndarray
    label : str
    min_effectsize : float
        The minimum effect size of a gene for inclution in returned DE.
        Default 2.  Overriden when for_gsea=True.
    FDR : float
        Maximum passing gene FDR.  Default 0.01.  Overriden when
        for_gsea=True.
    aux : list of of PopulationStat instances
        Additional counts to merge into the outgroup.  Useful for
        cross-sample comparisons.
    for_gsea : bool
        Format for results for gsea. overrides min_effectsize and FDR to
        automatically return all genes if true.  Default False
    garbage_collect : bool
        Manually trigger garbage collection between cluster binomial test
        runs.  Default False.
    verbose : bool
        Print progress statements.


    Returns
    -------
    up : pandas Dataframe
        Dataframe of upregulated genes in cluster, with colums: ['cluster',
        'ens', 'gene','ncluster','nrest','fdr','log2_effect'], where
        ncluster is the number of cells expressing in the cluster, and nrest
        is the number of cells expressing in the outgroup.
    down : pandas Dataframe
        Dataframe of downregulated genes in cluster. Columns same as up.
    cluster_info : pandas Dataframe
        Dataframe of total number of cells in cluster (column n_cells_cluster)
        and the outgroup (column n_cells_rest).

    """
    if for_gsea:
        min_effectsize = 0
        FDR = 1.01
        min_proportion = 0

    up, down, cluster_info  = [], [], []
    # cycle through clusters
    cmg = cluster_mask_generator(clusters)
    for c,cluster_mask in cmg:
        cluster_exp = expression[expression.columns[cluster_mask]]
        rest_exp  = expression[expression.columns[~cluster_mask]]
        cluster_id = '{}.{}'.format(population_id, c)
        cluster = PopulationStats.create_from_expression(cluster_id,
                cluster_exp)
        rest_id = '{}.not{}'.format(population_id, c)
        rest = PopulationStats.create_from_expression(rest_id, rest_exp)

        if verbose:
            message = 'Calculating diffex for %s vs. %s' % (cluster.id, rest.id)
            print(message)
        # merge aux, make sure it's a copy with [:] operator
        if len(aux):
            rest.merge(aux[:], inplace=True)
        # rest.sort(cluster.n_cells_exp.index)

        cluster_info.append( [cluster.id, cluster.n_cells, rest.n_cells] )
        up_c, down_c = binomial_test(ingroup=cluster, outgroup=rest,
                min_effectsize=min_effectsize, FDR=FDR,
                min_proportion=min_proportion)
        up_c['cluster'] = cluster.id
        up_c['gene'] = genes.loc[up_c.index].gene
        up_c['ens'] = genes.loc[up_c.index].ens
        up.append(up_c)

        down_c['cluster'] = cluster.id
        down_c['gene'] = genes.loc[down_c.index].gene
        down_c['ens'] = genes.loc[down_c.index].ens
        down.append(down_c)

        if garbage_collect:
            gc.collect()

    # compile results
    order = ['cluster','ens','gene','ncluster','nrest','fdr','log2_effect']
    df_up = pd.concat(up).reset_index()
    df_down = pd.concat(down).reset_index()
    cluster_info = pd.DataFrame(np.array(cluster_info),
            columns=['cluster', 'n_cells_cluster', 'n_cells_rest'])
    return df_up[order], df_down[order], cluster_info


def write_diffex(up, down, outdir, label):
    """
    Write the results of differential expression to two files
    Passing empty list to up or down will avoid writing anything
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if len(up):
        up_path = '{}/{}.up.tsv'.format(outdir, label)
        up.to_csv(up_path, header=True, index=False, sep='\t')
    if len(down):
        down_path = '{}/{}.down.tsv'.format(outdir, label)
        down.to_csv(down_path, header=True, index=False, sep='\t')


def write_diffex_by_cluster(up, down, outdir, cluster_info):
    clusters = np.union1d(up.cluster.unique(), down.cluster.unique())
    for cluster in clusters:
        up_c = up.loc[up.cluster == cluster]
        down_c = down.loc[down.cluster == cluster]
        nCells = cluster_info.loc[cluster_info.cluster==cluster,
                'n_cells_cluster'].values[0]
        nRest = cluster_info.loc[cluster_info.cluster==cluster,
                'n_cells_rest'].values[0]
        label_c = '{}.nCells_{}.nRest_{}'.format(cluster, nCells, nRest)
        write_diffex(up_c, down_c, outdir, label_c)


def diffex_heatmap(expression, genes, clusters, up, ntop, outdir, label,
        fdr_cutoff=0.01, normed=False):
    """
    Generates gene expression heatmap of the top differentially specific
    genes for each cluster.

    Parameters
    ----------
    expression : DataFrame
        DataFrame of expression counts
    genes : DataFrame
        Two column DataFrame of gene names
    up : DataFrame
        The up DataFrame from binomial_test_cluster_vs_rest
    ntop : int
        The number of top genes to use to create the heatmap. If a gene was
        also a top gene for a previous cluster, the next highest effect size
        gene with a corrected p-value < fdr_cuttoff is used.
    outdir : str
        Output directy for pdf
    label : str
        Prefix to prepend to the filename
    fdr_cutoff : float, optional (Default: 0.01)
        Do not include genes in heatmap that are most differentially specific
        with fdr_bh corrected p values >= this value, even if they are in the
        top `ntop`.

    """
    nclusters = len(np.unique(clusters))
    # relable unclustered cells (assumed labeled with -1)
    replace_neg1_with = -1
    if -1 in np.unique(clusters):
        replace_neg1_with = np.max(clusters) + 1
        clusters = clusters.copy()
        clusters[clusters == -1] = replace_neg1_with

    expression = expression.set_index(genes.ens)

    # get top (significantly) differentially specific genes per cluster
    top_genes, top_gene_names = [], []
    for c in np.sort(np.unique(clusters)):
        if c == replace_neg1_with:
            c = -1
        my_diffex = up[up.cluster.str.split('.').str[-1].astype(int) == c
                ].sort_values(by=['fdr', 'log2_effect'], ascending=[True,False])
        # filter gene names
        gene_name_mask = ~my_diffex.gene.str.contains('-')
        # only look at significant genes
        fdr_mask = my_diffex.fdr <= fdr_cutoff
        # don't add genes already on list
        unused_mask = ~my_diffex.ens.isin(top_genes)
        my_diffex = my_diffex[gene_name_mask & fdr_mask & unused_mask]
        top_genes.extend(my_diffex.head(ntop).ens.tolist())
        top_gene_names.extend(my_diffex.head(ntop).gene.tolist())


    # get cells with mergesort (a stable sort)
    cell_order = np.argsort(clusters, kind='mergesort')
    if not normed: # if not already normalized, normalized expression
        expression = np.log2(expression / expression.sum(axis=0) * 1e4 + 1)

    diffex_matrix = expression.loc[top_genes][cell_order]
    diffex_matrix.index = top_gene_names


    # plot heatmap of gene expression for top_N genes ordered by cluster
    # assignment
    outfile ='{}/{}pg.diffex.pdf'.format(outdir,label+'.' if len(label) else '')
    mpl, plt, _ = _import_plotlibs()
    colors = get_cluster_cmap(nclusters, mpl)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(outfile) as pdf:
        fig,ax = plt.subplots()
        L = float(diffex_matrix.shape[0])/100.*15.
        fig.set_size_inches(20,L)
        heatmap = ax.pcolor(diffex_matrix.values,cmap='BuGn')
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_yticks(np.arange(diffex_matrix.shape[0])+0.5,minor=False)
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=9)
        labels = diffex_matrix.index.tolist()
        ax.set_yticklabels(labels,minor=False)
        pdf.savefig()
        plt.close()
        fig,ax=plt.subplots()
        fig.set_size_inches(20,1)
        cMap = mpl.colors.ListedColormap(colors)
        clusterids = clusters[cell_order]

        # clusterids = [0 for pt in range(clusters.count(0))]
		# for i in range(1,Nclusters):
			# clusterids.extend([i for pt in range(clusters.count(i))])
        heatmap = ax.pcolor([clusterids,clusterids],cmap=cMap)
        fig = plt.gcf()
        ax = plt.gca()
        pdf.savefig()
        plt.close()
	# return 0

