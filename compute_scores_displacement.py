import os
import numpy as np
from scipy.stats import norm
import xarray as xr
from sklearn.metrics import pairwise_distances

# Imports for scores
from scores.probability import crps_cdf, crps_for_ensemble
from scoringrules import energy_score
from utils import ensmean_fraction_skill_score

from utils import read_case

import pandas as pd

np.random.seed(0)
in_dir = "fields/"
out_dir = "data/"
cache_dir = "data/"

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

all_shifts = np.arange(1, 51, 1)
n_members = 1000

comparison_dict = {
    'D02': ['P10'] + ['P10_{:02d}'.format(i) for i in all_shifts] + ['P00'],
}

df = pd.DataFrame({"obs_fcst": [], "score_name": [], 'score_value': []})

# ---------------------------- Spatial dependence ---------------------------- #

def nearest_psd_clip(A):
    A = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, 0, None)
    return (vecs * vals) @ vecs.T

distance = pairwise_distances(np.arange(101).reshape(-1,1))

# Exponentials
d1 = 2000.0
d2 = 30.0
correlation_exp = np.exp(-distance/d1)
correlation_exp2 = np.exp(-distance/d2)

# Exponential with cutoff
cutoff = 5
correlation_cutoff = correlation_exp2 * (distance < cutoff)
correlation_cutoff = nearest_psd_clip(correlation_cutoff)

# ----------------------- Generate correlated uniform ----------------------- #
Z_exp_np = np.random.multivariate_normal(mean=np.zeros(101), cov=correlation_exp, size=n_members)
Z_cutoff_np = np.random.multivariate_normal(mean=np.zeros(101), cov=correlation_cutoff, size=n_members)

u_exp_np = norm.cdf(Z_exp_np)
u_cutoff_np = norm.cdf(Z_cutoff_np)

# Convert to xarray DataArrays
U_exp = xr.DataArray(
    u_exp_np,
    dims=['member', 'dim_x'],
    coords={'member': np.arange(n_members), 'dim_x': np.arange(101)}
)
U_cutoff = xr.DataArray(
    u_cutoff_np,
    dims=['member', 'dim_x'],
    coords={'member': np.arange(n_members), 'dim_x': np.arange(101)}
)

# ---------------------------------------------------------------------------- #
new_rows = {"obs_fcst": [], "score_name": [], 'score_value': []}

for obs_name, fcst_list in comparison_dict.items():
    obs,x,value = read_case(obs_name, in_dir)
    obs = xr.DataArray(np.argmax(obs, axis=0), dims=['dim_x'], coords={'dim_x': x})

    for fcst_name in fcst_list:
        print(f'{obs_name} - {fcst_name}')
        fcst,x,value = read_case(fcst_name, in_dir)
        fcst_cdf = np.cumsum(fcst, axis=0)
        fcst_cdf = np.where(fcst_cdf > 1, 1.0, fcst_cdf)
        fcst_cdf = xr.DataArray(fcst_cdf, dims=['dim_value', 'dim_x'], coords={'dim_value': value, 'dim_x': x})

        fcst = xr.DataArray(fcst, dims=['dim_value', 'dim_x'], coords={'dim_value': value, 'dim_x': x})
        
        # generate ensembles using xarray
        ens_exp = xr.apply_ufunc(
            lambda cdf, uniform: np.searchsorted(cdf, uniform, side="left"),
            fcst_cdf,  # CDF values along dim_value for each dim_x
            U_exp,  # Uniform values for each member and dim_x
            input_core_dims=[['dim_value'], []],
            vectorize=True,
            output_dtypes=[int]
        )
        ens_cutoff = xr.apply_ufunc(
            lambda cdf, uniform: np.searchsorted(cdf, uniform, side="left"),
            fcst_cdf,
            U_cutoff,
            input_core_dims=[['dim_value'], []],
            vectorize=True,
            output_dtypes=[int]
        )
        
        # CRPS
        crps_prob = crps_cdf(fcst_cdf, obs, threshold_dim='dim_value', preserve_dims=("dim_x"), integration_method='trapz').total.values
        crps_ens_exp = crps_for_ensemble(ens_exp, obs, ensemble_member_dim="member", method="fair", preserve_dims=("dim_x")).values
        crps_ens_cutoff = crps_for_ensemble(ens_cutoff, obs, ensemble_member_dim="member", method="fair", preserve_dims=("dim_x")).values
        new_rows['obs_fcst'] += [f'{obs_name} - {fcst_name}']*3
        new_rows['score_name'].append('crps_prob')
        new_rows['score_value'].append(np.mean(crps_prob))
        new_rows['score_name'].append('crps_exp')
        new_rows['score_value'].append(np.mean(crps_ens_exp))
        new_rows['score_name'].append('crps_cutoff')
        new_rows['score_value'].append(np.mean(crps_ens_cutoff))
        
        
        # MSE
        mean_prob = fcst.T @ fcst.dim_value
        mean_exp = ens_exp.mean(dim='member')
        mean_cutoff = ens_cutoff.mean(dim='member')
        mse_prob = (mean_prob.values - obs)**2
        mse_ens_exp = (mean_exp - obs)**2
        mse_ens_cutoff = (mean_cutoff - obs)**2
        new_rows['obs_fcst'] += [f'{obs_name} - {fcst_name}']*3
        new_rows['score_name'].append('mse_prob')
        new_rows['score_value'].append(np.mean(mse_prob.values))
        new_rows['score_name'].append('mse_exp')
        new_rows['score_value'].append(np.mean(mse_ens_exp.values))
        new_rows['score_name'].append('mse_cutoff')
        new_rows['score_value'].append(np.mean(mse_ens_cutoff.values))
        
        # Energy score
        es_exp = energy_score(obs.values, ens_exp.values, m_axis = -1, v_axis = -2)
        es_cutoff = energy_score(obs.values, ens_cutoff.values, m_axis = -1, v_axis = -2)
        new_rows['obs_fcst'] += [f'{obs_name} - {fcst_name}']*2
        new_rows['score_name'].append('es_exp')
        new_rows['score_value'].append(np.mean(es_exp))
        new_rows['score_name'].append('es_cutoff')
        new_rows['score_value'].append(np.mean(es_cutoff))
        
        # emFSS
        thresholds = [45,55]
        neighborhood_size = 10
        for threshold in thresholds:
            fss_prob = ensmean_fraction_skill_score(obs, mean_prob, threshold=threshold, window_size=neighborhood_size)
            fss_exp = ensmean_fraction_skill_score(obs, mean_exp, threshold=threshold, window_size=neighborhood_size)
            fss_cutoff = ensmean_fraction_skill_score(obs, mean_cutoff, threshold=threshold, window_size=neighborhood_size)
            new_rows['obs_fcst'] += [f'{obs_name} - {fcst_name}']*3
            new_rows['score_name'].append('fss_prob_t' + str(threshold))
            new_rows['score_value'].append(np.mean(fss_prob))
            new_rows['score_name'].append('fss_exp_t' + str(threshold))
            new_rows['score_value'].append(np.mean(fss_exp))
            new_rows['score_name'].append('fss_cutoff_t' + str(threshold))
            new_rows['score_value'].append(np.mean(fss_cutoff))

        
new_rows = pd.DataFrame(new_rows)
df = pd.concat([df, new_rows], ignore_index=True)

df.to_csv(out_dir + 'scores_displacement.csv', index=False)