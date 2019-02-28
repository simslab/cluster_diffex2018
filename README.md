# cluster_diffex

Representative marker selection, cluster, visualization, and binomial differential expression pipeline reported in [Yuan *et al.* 2018](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-018-0567-9), [Mizrak *et al.* 2019](https://www.sciencedirect.com/science/article/pii/S2211124718319740?via%3Dihub), [Levitin *et al.* 2019](https://onlinelibrary.wiley.com/doi/full/10.15252/msb.20188557) and [Szabo, Levitin *et al.* 2019](https://www.biorxiv.org/content/10.1101/555557v1).

The pipeline comes in two flavors: marker selection with fixed bin widths (as in Yuan *et al* 2018 and Mizrak *et al.* 2019) or an updated procedure with rolling windows and a scaled drop out score (Levitin *et al.* 2019 and Szabo, Levitin *et al.* 2019).

# Installation
### Environment & Dependencies
This pipline requires Python >= 3.6 and the packages:
- scipy >= 1.1
- numpy
- pandas
- statsmodels
- matplotlib
- seaborn
- umap
- phenograph == 1.5.2 (https://github.com/jacoblevine/PhenoGraph)

Optional:
- [dmaps](https://github.com/hsidky/dmaps) 


The easiest way to setup a python environment for s with [anaconda](https://www.continuum.io/downloads).
```
conda create -n cluster_diffex_p36 python=3.6 scipy=1.1 pandas statsmodels matplotlib seaborn

# older anaconda
source activate cluster_diffex_p36
# XOR newer anaconda
conda activate cluster_diffex_p36

# Install UMAP 
conda install -c conda-forge umap-learn

# Install phenograph
pip install git+https://github.com/jacoblevine/phenograph.git
```

#### Optional dependencies
Optionally, install dmaps. First make sure you have cmake.  On debian-based distributions do:
```
sudo apt install cmake
```
XOR on OSX with homebrew do: 
```
brew install cmake
```
XOR something else for a different OS/package manager.  Then:
```
pip install git+https://github.com/hsidky/dmaps.git
```
### Installing the code 
Once you have set up the environment, clone this repository and install.
```
git clone git@github.com:simslab/cluster_diffex2018.git
pip install ./cluster_diffex2018
```
## Running the pipeline
A typical run of the pipeline might look like:
```
python scripts/cluster_diffex.py --count-matrix UMI_MATRIX.txt -o OUTDIR  -p PREFIX
```
where UMI_MATRIX.txt is a whitespace delimited gene by cell UMI count matrix with two leading columns of gene attributes: <pre>ENSEMBL_ID  GENE_NAME  UMICOUNT_CELL0  UMICOUNT_CELL1 ... </pre>.

To see more options, such as setting `k` for the k nearest neighbors graph, using a list of markers in a file, or using the older scoring scheme:
```
python scripts/cluster_diffex.py -h
```
### Output files (standard analysis)
- `OUTDIR/PREFIX.dropout_curve.pdf` The dropout curve with marker genes colored in green

- `OUTDIR/PREFIX.dropout_threshold.txt` A record of the absolute threshold ('t') and the adaptive threshold. The minimum is used as the cutoff for marker selection.

- `OUTDIR/PREFIX.marker_ix.txt` Indices of the marker gene rows used in the original count matrix.

- `OUTDIR/PREFIX.markers.txt` ENSEMBL\_ID and GENE\_NAME for selected genes.

- `OUTDIR/PREFIX.markers.dcorrSP.umap.txt` Coordinates for UMAP embedding of cells (determined using Spearman's correlation distace on marker genes).

- `OUTDIR/PREFIX.markers.dcorrSP.umap.pdf` Plot of UMAP embedding of cells (determined using Spearman's correlation distance on marker genes).

- `OUTDIR/PREFIX.markers.dcorrSP.pg.txt` Integer labels for phenograph clusters (determind using Spearman's correlation distance on marker genes). -1 indicates an unclustered cell.

- `OUTDIR/PREFIX.markers.dcorrSP.pg.info.txt` Number of neighbors used for clustering (k) and final modularity of the clustering (Q).

- `OUTDIR/PREFIX.markers.dcorrSP.umap.pg.pdf` Plot of UMAP embedding of cells (determined using Spearman's correlation distace) colored by Phenograph cluster.

- `OUTDIR/PREFIX.markers.dcorrSP.CLUSTER_ID.nCells_N.nRest_M.up.tsv` For a cluster CLUSTER\_ID with N cells in the cluster, and M cells in the rest of the dataset, a table of gene ids, names, count in cluster, count out of cluster, fdr_bh corrected pvalues and effect sizes by a binomial test for binary upregulation. Ordered by effect size (decreasing).

- `OUTDIR/PREFIX.markers.dcorrSP.CLUSTER_ID.nCells_N.nRest_M.down.tsv` For a cluster CLUSTER\_ID with N cells in the cluster, and M cells in the rest of the dataset, a table of gene ids, names, count in cluster, count out of cluster, fdr_bh corrected pvalues and effect sizes by a binomial test for binary downregulation. Ordered by negative effect size.

- `OUTDIR/PREFIX.markers.dcorrSP.pg.diffex.pdf` Heatmap of normalized expression for the the top differentially expressed genes (by a binomial test) in each cluster.

## Optional: install as a command line utility
In `setup.py`, uncomment the line:
```
# scripts=['scripts/cluster_diffex']
```
Then reinstall the package. While in the base directory:
```
pip install .
```
All commands run with the 'python cluster_diffex.py' can then just be run with 'cluster_diffex' instead.
