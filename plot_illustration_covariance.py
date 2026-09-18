import os
import string

import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MultipleLocator

from utils import read_case

in_dir = 'fields/'
cache_dir = 'data/'
out_dir = 'figures/'
os.makedirs(out_dir, exist_ok=True)

CMAP_B = matplotlib.colors.LinearSegmentedColormap.from_list(
    "rb_cmap_blue", ["white", "#1f6fb3"], 512
)
CMAP_R = matplotlib.colors.LinearSegmentedColormap.from_list(
    "rb_cmap_red", ["white", (1.0, 0.3, 0.3)], 512
)

EXPERIMENTS_DICT = {
    "Illustration": {
        "obs": "D02",
        "fcsts": ["P04", "P06"],
    },
}


def load_uniform_draws(sample_file: str, n_members: int | None = None):
    if not os.path.exists(sample_file):
        raise FileNotFoundError(
            f"Missing shared sample file: {sample_file}. Run metrics.py first to create it."
        )

    with np.load(sample_file) as data:
        u_exp_np = data["u_exp"]
        u_exp2_np = data["u_exp2"]
        u_cutoff_np = data["u_cutoff"]

    if n_members is not None:
        total_members = u_exp_np.shape[0]
        if n_members > total_members:
            raise ValueError(
                f"Requested {n_members} members, but only {total_members} are available in {sample_file}."
            )
        u_exp_np = u_exp_np[:n_members, :]
        u_exp2_np = u_exp2_np[:n_members, :]
        u_cutoff_np = u_cutoff_np[:n_members, :]

    n_x = u_exp_np.shape[1]
    members = np.arange(u_exp_np.shape[0])
    dim_x = np.arange(n_x)

    u_exp = xr.DataArray(u_exp_np, dims=["member", "dim_x"], coords={"member": members, "dim_x": dim_x})
    u_exp2 = xr.DataArray(u_exp2_np, dims=["member", "dim_x"], coords={"member": members, "dim_x": dim_x})
    u_cutoff = xr.DataArray(u_cutoff_np, dims=["member", "dim_x"], coords={"member": members, "dim_x": dim_x})

    return u_exp, u_exp2, u_cutoff


def sample_ensemble_from_cdf(fcst_cdf: xr.DataArray, uniforms: xr.DataArray, fcst_name: str) -> xr.DataArray:
    # Keep spatial dependence consistent when forecast names include horizontal mirroring.
    aligned_uniforms = uniforms.sel(dim_x=fcst_cdf.dim_x)
    if "h" in fcst_name:
        aligned_uniforms = aligned_uniforms.isel(dim_x=slice(None, None, -1))

    cdf_np = fcst_cdf.transpose("dim_x", "dim_value").values
    uniforms_np = aligned_uniforms.transpose("member", "dim_x").values

    n_members, n_x = uniforms_np.shape
    sampled = np.empty((n_members, n_x), dtype=int)
    for i in range(n_x):
        sampled[:, i] = np.searchsorted(cdf_np[i, :], uniforms_np[:, i], side="left")

    if "h" in fcst_name:
        sampled = sampled[:, ::-1]

    return xr.DataArray(
        sampled,
        dims=["member", "dim_x"],
        coords={"member": aligned_uniforms.member.values, "dim_x": fcst_cdf.dim_x.values},
    )


def apply_selected_comparison_style(ax):
    ax.set_xlim(left=-0.5, right=100.5)
    ax.set_ylim(bottom=-0.5, top=100.5)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(top=True, right=True, which="both")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlabel(r"$x$ (spatial coordinate)")
    ax.set_ylabel("Variable value")


def plot_experiment_grid(experiment_name: str, fcst_names: list[str], u_exp, u_exp2, u_cutoff):
    n_rows = len(fcst_names)
    col_titles = ["Prob.", r'Exp. ($\lambda=2000$)', r'Exp. ($\lambda=30$)', "Cut-off"]
    fig, axes = plt.subplots(
        n_rows,
        len(col_titles),
        figsize=(5*len(col_titles), 5*n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=16, pad=12)
        axes[1, j].set_title(title, fontsize=16, pad=12)

    for i, fcst_name in enumerate(fcst_names):
        print(f"{experiment_name}: {fcst_name}")
        fcst_pdf,x,value = read_case(fcst_name, in_dir)
        fcst_cdf = np.cumsum(fcst_pdf, axis=0)
        fcst_cdf = np.where(fcst_cdf > 1, 1.0, fcst_cdf)
        fcst_cdf = xr.DataArray(fcst_cdf, dims=['dim_value', 'dim_x'], coords={'dim_value': value, 'dim_x': x})
        fcst_pdf = xr.DataArray(fcst_pdf, dims=['dim_value', 'dim_x'], coords={'dim_value': value, 'dim_x': x})

        vmax_temp = float(fcst_pdf.max().values) * 1.1
        if float(fcst_pdf.mean().values) > vmax_temp * 0.5:
            vmax_temp = vmax_temp * 3

        line_color = CMAP_B(0.85)
        x_values = fcst_pdf.x.values if "x" in fcst_pdf.coords else fcst_pdf.dim_x.values

        ens_exp = sample_ensemble_from_cdf(fcst_cdf, u_exp, fcst_name)
        ens_exp2 = sample_ensemble_from_cdf(fcst_cdf, u_exp2, fcst_name)
        ens_cutoff = sample_ensemble_from_cdf(fcst_cdf, u_cutoff, fcst_name)

        # Column 1: forecast marginal PDF map
        img_extent = (-0.5, 100.5, -0.5, 100.5)
        axes[i, 0].imshow(
            fcst_pdf.transpose("dim_value", "dim_x").values,
            cmap=CMAP_B,
            vmin=0,
            vmax=vmax_temp,
            interpolation="nearest",
            origin="lower",
            extent=img_extent,
        )
        axes[i, 0].set_xlabel(r"$x$ (spatial coordinate)")
        axes[i, 0].set_ylabel("Variable value")
        axes[i, 0].set_title("Prob.", fontsize=16, pad=12)
        apply_selected_comparison_style(axes[i, 0])

        fcst_description = fcst_pdf.attrs.get("description", "")
        axes[i, 0].text(
            -0.2, 0.5,
            r"$\bf{" + fcst_name.replace("_", r"\_") + "}$: " + fcst_description,
            transform=axes[i, 0].transAxes, ha='center', va='center', fontsize=16,
            rotation=90,
        )

        # Columns 2-4: ensemble trajectories
        for j, ensemble in enumerate([ens_exp, ens_exp2, ens_cutoff], start=1):
            ensemble_np = ensemble.values
            for member_idx in range(ensemble_np.shape[0]):
                axes[i, j].plot(
                    x_values,
                    ensemble_np[member_idx, :],
                    alpha=0.2,
                    color=line_color,
                    linewidth=2,
                )
            axes[i, j].plot(
                x_values,
                ensemble_np.mean(axis=0),
                color="black",
                linewidth=1.5,
            )
            
            apply_selected_comparison_style(axes[i, j])

    for idx, ax in enumerate(axes.flat):
        prefix = "(" + string.ascii_lowercase[idx] + ") "
        ax.set_title(prefix + ax.get_title(), fontsize=16, pad=12)

    output_path = out_dir + f"{experiment_name.lower()}_covariance.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


u_exp, u_exp2, u_cutoff = load_uniform_draws(cache_dir + 'uniform_samples.npz', n_members=50)

print(u_exp.shape, u_cutoff.shape)

for experiment_name, experiment_info in EXPERIMENTS_DICT.items():
    plot_experiment_grid(
        experiment_name=experiment_name,
        fcst_names=experiment_info["fcsts"],
        u_exp=u_exp,
        u_exp2=u_exp2,
        u_cutoff=u_cutoff,
    )

