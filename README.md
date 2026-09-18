# 1D Idealized Probabilistic Fields

**Zenodo** : [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXX)

Data and code of "A dataset of one-dimensional idealized probabilistic fields" by G. Skok and R. Pic ([arxiv:]())

## Citation

```bibtex

```

## Organization of the repository

`pyproject.toml` contains information on the dependencies to run the code in this repository. Using `uv`, you simply need to run `uv sync` to set it up.

## 1D Fields

* `create_idealized_fields_and_save_to_NetCDF.py` : code to create the NetCDF files of the 1D fields
* `fields.zip` :  zip of the already generated 1D fields

### Selected comparisons

* Visualization (Figure 4): `plot_selected_comparisons.py`
* Computation of scores : `compute_scores_selected_comparisons.py`
* Illustration of the choice of covariance (Figure 5) : `plot_illustration_covariance.py`

### Effect of displacement

* Computation of scores : `compute_scores_displacement.py`
* Visualization (Figure 6): `plot_displacement.py`

## Related resources

* [Bridging The Gap](https://pic-romain.github.io/bridging-the-gap/)