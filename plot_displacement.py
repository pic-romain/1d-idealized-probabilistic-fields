import string

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

in_dir = "data/"
out_dir = "figures/"

df = pd.read_csv(in_dir + 'scores_displacement.csv')

from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,
    'savefig.dpi': 300,
})

lw = 2
lw_ref = 1
marker_size = 4
legend_elements = [
    Line2D([0], [0], color='C0', lw=lw, ls='solid', marker='o', label='CRPS'),
    Line2D([0], [0], color='C1', lw=lw, ls='solid', marker='o', label='MSE'),
    Line2D([0], [0], color='gray', lw=lw, ls='solid', label='Prob.'),
    Line2D([0], [0], color='gray', lw=lw, ls='dashed', label=r'Exp. ($\lambda=2000$)'),
    Line2D([0], [0], color='gray', lw=lw, ls='dotted', label='Cut-off'),
                   ]


all_shifts = list(range(0, 51))
obs_name = 'D02'
fcst_list = ['P10']+[f'P10_{i:02d}' for i in all_shifts[1:]]

df_pivot = df.pivot(index='obs_fcst', columns='score_name', values='score_value')
ls_ens = {'Prob.': 'solid', r'Exp. ($\lambda=2000$)': 'dashed', 'Cut-off': 'dotted'}
color_score = {'Agg. CRPS': 'C0', 'MSE': 'C1', 'ES': 'C2', 'emFSS (t=45)': 'red', 'emFSS (t=55)': 'purple'}

column_groups = [
    ('MSE',  [('mse_prob','Prob.'),  ('mse_exp',r'Exp. ($\lambda=2000$)'),  ('mse_cutoff','Cut-off')]),
    ('Agg. CRPS', [('crps_prob','Prob.'), ('crps_exp',r'Exp. ($\lambda=2000$)'), ('crps_cutoff','Cut-off')]),
    ('ES',   [('es_exp',r'Exp. ($\lambda=2000$)'), ('es_cutoff','Cut-off')]),
]

def apply_publication_style(ax):
    ax.tick_params(top=True, right=True, which='both', direction='out')
    ax.set_xlim(left=0, right=51)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


n_groups = len(column_groups)
fig, axs = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5), squeeze=False)
axs = axs[0]
for p, (score_group, items) in enumerate(column_groups):
    ax = axs[p]
    print(f"Score: {score_group}")
    scores_dict = {}
    for score_name, pretty_name in items:
        if score_name not in scores_dict.keys():
            scores_dict[score_name] = {'name': pretty_name, 'values': []}
        if 'es' in score_name:
            score_name_exp = score_name.replace('cutoff', 'exp')
        else:
            score_name_exp = score_name.replace('cutoff', 'prob').replace('exp', 'prob')
        ref = df.loc[(df['obs_fcst'] == f'{obs_name} - P10') & (df['score_name'] == score_name_exp), 'score_value'].values[0]
        for i in range(len(fcst_list)):
            obs_fcst = f'{obs_name} - {fcst_list[i]}'
            value = df.loc[(df['obs_fcst'] == obs_fcst) & (df['score_name'] == score_name), 'score_value'].values[0]
            scores_dict[score_name]['values'].append(
                (value - ref) / ref
            )

        flat_value = df.loc[(df['obs_fcst'] == f'{obs_name} - P00') & (df['score_name'] == score_name), 'score_value'].values[0]
        scores_dict[score_name]['value_P00'] = (flat_value - ref) / ref
    for score_name in scores_dict.keys():
        ax.plot(all_shifts, scores_dict[score_name]['values'], color = color_score[score_group], linestyle=ls_ens[scores_dict[score_name]['name']], linewidth=lw)
        ax.hlines(y=scores_dict[score_name]['value_P00'], xmin=all_shifts[1], xmax=all_shifts[-1], color=color_score[score_group], linestyle=ls_ens[scores_dict[score_name]['name']], linewidth=lw_ref)
    ax.set_title(f'({string.ascii_lowercase[p]}) {score_group}', pad=10)
    ax.set_xlabel('Displacement')
    ax.set_ylabel('Relative score')
    apply_publication_style(ax)

    # Only advertise the correlation structures actually drawn in this panel.
    panel_legend = [
        Line2D([0], [0], color='gray', lw=lw, ls=ls_ens[pretty_name], marker='o',
               markersize=marker_size, label=pretty_name)
        for _, pretty_name in items
    ]
    panel_legend.append(
        Line2D([0], [0], color='gray', lw=lw_ref, ls='solid', label='P00')
    )
    ax.legend(handles=panel_legend, loc='upper left', framealpha=1, edgecolor='black')

plt.tight_layout()
plt.savefig(out_dir + 'displacement_effect_on_scores.png', dpi=300, bbox_inches='tight')