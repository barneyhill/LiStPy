# LiStPy
## Minimal and vectorised Li and Stephens implementation in PyTorch.

This library contains a minimal and vectorised implementation of the Li and Stephens model (haploid, forwards-backwards) as described in the [2022 `kalis` paper](https://arxiv.org/abs/2212.11403). 

## Goals
- A simple formulation of the Li and Stephens model to showcase both the model itself and the PyTorch library.
- A general platform to investigate O(N^2) -> O(N) optimisations.
- Investigating automatic differentation of model parameters with respect to imputation/phasing accuracy.
- GPU support.

## TODO
- Testing

## Example

see notebook.ipynb
