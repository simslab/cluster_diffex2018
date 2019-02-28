#!/usr/bin/env python

from setuptools import find_packages, setup

__version__ = '0.0.1'

requires = ['scipy >= 1.1',
            'numpy',
            'pandas',
            'statsmodels',
            'phenograph == 1.5.2',
            'umap-learn',
            ]

extras_require = {'diffusion embeddings' : 'dmaps'}

setup(
    name='clusterdiffex',
    version=__version__,
    # scripts=['scripts/cluster_diffex'],
    python_requires='>=3.6',
    install_requires=requires,
    extras_require=extras_require,
    author = 'Sims Lab',
    author_email = '',
    description='Representative code for clustering, binomial differential expression and visualization',
    license="MIT",
    packages=find_packages(),
)

