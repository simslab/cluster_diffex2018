# cluster_diffex

Representative marker selection, cluster, visualization, and binomial differential expression pipeline reported in [Yuan *et al.* 2018](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-018-0567-9), [Mizrak *et al.* 2019](https://www.sciencedirect.com/science/article/pii/S2211124718319740?via%3Dihub), [Levitin *et al.* 2019](https://onlinelibrary.wiley.com/doi/full/10.15252/msb.20188557) and [Szabo, Levitin *et al.* 2019](https://www.biorxiv.org/content/10.1101/555557v1).

The pipeline comes in two flavors: marker selection with fixed bin widths (as in Yuan *et al* 2018 and Mizrak *et al.* 2019) or an updated procedure with rolling windows and a scaled drop out score (Levitin *et al.* 2019 and Szabo, Levitin *et al.* 2019).

# Installation
### Environment & Dependencies
This pipline requires Python >= 3.6 and the packages:
- scipy >= 1.1
- numpy
- pandas
- scikit-learn
- statsmodels
- matplotlib
- seaborn
- [umap-learn](https://github.com/lmcinnes/umap)
- [phenograph](https://github.com/jacoblevine/PhenoGraph) == 1.5.2 

Optional:
- [dmaps](https://github.com/hsidky/dmaps) 


The easiest way to setup a python environment for cluster_diffex is with [anaconda](https://www.continuum.io/downloads).
```
conda create -n cluster_diffex_p36 python=3.6 scikit-learn statsmodels seaborn

# older anaconda
source activate cluster_diffex_p36
# XOR newer anaconda
conda activate cluster_diffex_p36

# Install UMAP 
conda install -c conda-forge umap-learn

# Install phenograph
pip install git+https://github.com/jacoblevine/phenograph.git
```

#### Installing optional dependencies
1. Optionally, install loompy
```
pip install -U loompy
```

2. Optionally, install dmaps. First make sure you have cmake.  On debian-based distributions do:
```
sudo apt install cmake
```
XOR on OSX with homebrew do: 
```
brew install cmake
```
XOR something else for a different OS/package manager.  

Then, install dmaps:
```
pip install git+https://github.com/hsidky/dmaps.git
```
### Installing the code 
Once you have set up the environment, clone this repository and install.
```
git clone git@github.com:simslab/cluster_diffex2018.git
cd cluster_diffex2018
pip install .
```
## Running the pipeline
A typical run of the pipeline might look like:
```
python scripts/cluster_diffex.py --count-matrix UMI_MATRIX.txt -o OUTDIR  -p PREFIX
```
where UMI_MATRIX.txt is a whitespace delimited gene by cell UMI count matrix with two leading columns of gene attributes: `ENSEMBL_ID  GENE_NAME  UMICOUNT_CELL0  UMICOUNT_CELL1 ... `.

To see more options, such as setting `k` for the k nearest neighbors graph, using a preselected list of markers in a file, setting thresholds for marker gene selection, using the older scoring scheme, visualization with tSNE or dmaps, etc.:
```
python scripts/cluster_diffex.py -h
```
In particular, to use the old scoring scheme, add the flag `--unscaled-score`.
 
### Output files (standard analysis)
- `OUTDIR/PREFIX.dropout_curve.pdf` The dropout curve with marker genes colored in green

- `OUTDIR/PREFIX.dropout_threshold.txt` A record of the absolute threshold ('t') and the adaptive threshold. The minimum is used as the cutoff for marker selection.

- `OUTDIR/PREFIX.marker_ix.txt` Indices of the marker gene rows used in the origninal count matrix *after* all rows with only zeros have been removed.

- `OUTDIR/PREFIX.markers.txt` ENSEMBL\_ID and GENE\_NAME for selected genes.

- `OUTDIR/PREFIX.markers.dcorrSP.umap.txt` Coordinates for UMAP embedding of cells (determined using Spearman's correlation distace on marker genes).

- `OUTDIR/PREFIX.markers.dcorrSP.umap.pdf` Plot of UMAP embedding of cells (determined using Spearman's correlation distance on marker genes).

- `OUTDIR/PREFIX.markers.dcorrSP.pg.txt` Integer labels for phenograph clusters (determind using Spearman's correlation distance on marker genes). -1 indicates an unclustered cell.

- `OUTDIR/PREFIX.markers.dcorrSP.pg.info.txt` Number of neighbors used for clustering (k) and final modularity of the clustering (Q).

- `OUTDIR/PREFIX.markers.dcorrSP.umap.pg.pdf` Plot of UMAP embedding of cells (determined using Spearman's correlation distace) colored by Phenograph cluster.

- `OUTDIR/PREFIX.markers.dcorrSP.CLUSTER_ID.nCells_N.nRest_M.up.tsv` For a cluster CLUSTER\_ID with N cells in the cluster, and M cells in the rest of the dataset, a table of gene ids, names, count in cluster, count out of cluster, fdr_bh corrected pvalues and effect sizes by a binomial test for binary upregulation. Ordered by effect size (decreasing).

- `OUTDIR/PREFIX.markers.dcorrSP.diffex_binom/PREFIX.markers.dcorrSP.CLUSTER_ID.nCells_N.nRest_M.down.tsv` For a cluster CLUSTER\_ID with N cells in the cluster, and M cells in the rest of the dataset, a table of gene ids, names, count in cluster, count out of cluster, fdr_bh corrected pvalues and effect sizes by a binomial test for binary downregulation. Ordered by negative effect size.

- `OUTDIR/PREFIX.markers.dcorrSP.diffex_binom/PREFIX.markers.dcorrSP.pg.diffex.pdf` Heatmap of normalized expression for the the top differentially expressed genes (by a binomial test) in each cluster.

- `OUTDIR/PREFIX.commandline_args.json` JSON file of command line arguments
