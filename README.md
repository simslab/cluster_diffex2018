# cluster-diffex

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
git clone https://github.com/hsidky/dmaps.git
pip install ./dmaps
```
### Installing the pipline code 
Once you have set up the environment, clone this repository and install.
```
git clone git@github.com:simslab/cluster_diffex2018.git
pip install ./cluster_diffex2018
```
