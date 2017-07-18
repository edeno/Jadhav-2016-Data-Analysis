import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from src.data_processing import read_netcdfs


def plot_power(path, group, brain_area, frequency, figsize=(15, 10),
               vmin=0.5, vmax=2):

    def transform_func(ds):
        return ds.sel(
            tetrode=ds.tetrode[ds.brain_area == brain_area],
            frequency=frequency
        )
    try:
        ds = read_netcdfs(path, dim='session', group=group,
                          transform_func=transform_func).power
    except ValueError:
        return
    DIMS = ['session', 'tetrode']

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    _plot_distribution(
        ds.isel(time=0), dims=DIMS, ax=axes[0, 0], color='midnightblue')
    axes[0, 0].set_title('Baseline Power')

    _plot_distribution(
        ds.isel(time=0), dims=DIMS, ax=axes[1, 0], color='midnightblue')
    axes[1, 0].set_title('Baseline Power')

    ds.mean(DIMS).plot(x='time', y='frequency', ax=axes[0, 1])
    axes[0, 1].set_title('Raw power')

    _plot_distribution(
        ds.sel(time=0.0, method='backfill'), dims=DIMS,
        ax=axes[1, 1], color='midnightblue')
    axes[1, 1].set_title('Raw power after ripple')

    (ds / ds.isel(time=0)).mean(DIMS).plot(
        x='time', y='frequency', ax=axes[0, 2],
        norm=LogNorm(vmin=vmin, vmax=vmax), cmap='RdBu_r',
        vmin=vmin, vmax=vmax, center=0)
    axes[0, 2].set_title('Change from baseline power')

    _plot_distribution(
        ds.sel(time=0.0, method='backfill') / ds.isel(time=0), dims=DIMS,
        ax=axes[1, 2], color='midnightblue')
    axes[1, 2].set_title('Change after ripple')
    axes[1, 2].axhline(1, color='black', linestyle='--')

    for ax in axes[0, 1:3]:
        ax.axvline(0, color='black', linestyle='--')

    plt.tight_layout()
    plt.suptitle(brain_area, fontsize=18, fontweight='bold')
    plt.subplots_adjust(top=0.85)


def plot_connectivity(
    path, brain_area_pair, frequency, resolution, covariate,
    level1, level2=None, figsize=(15, 10),
        connectivity_measure='coherence_magnitude'):

    def get_data(level):
        group = (
            '/'.join([resolution, covariate, level, connectivity_measure])
            .replace('//', '/'))

        def transform_func(ds):
            return ds.sel(
                tetrode1=ds.tetrode1[ds.brain_area1 == brain_area_pair[0]],
                tetrode2=ds.tetrode2[ds.brain_area2 == brain_area_pair[1]],
                frequency=frequency
            )
        return read_netcdfs(
            path, dim='session', group=group,
            transform_func=transform_func)[connectivity_measure]

    DIMS = ['session', 'tetrode1', 'tetrode2']
    try:
        ds1 = get_data(level1)
    except ValueError:
        return
    if level2 is not None:
        ds2 = get_data(level2)
    else:
        ds2 = ds1.isel(time=0)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    ds1.mean(DIMS).plot(
        x='time', y='frequency', ax=axes[0, 1], cmap='Greens')
    axes[0, 1].set_title(level1)

    _plot_distribution(
        ds1.sel(time=0.0, method='backfill'), dims=DIMS, ax=axes[1, 1],
        color='green')
    axes[1, 1].set_title('{level} after ripple'.format(level=level1))

    if level2 is not None:
        ds2.mean(DIMS).plot(
            x='time', y='frequency', ax=axes[0, 0], cmap='Purples')
        axes[0, 0].set_title(level2)
        _plot_distribution(ds2.sel(time=0.0, method='backfill'),
                           dims=DIMS, ax=axes[1, 0], color='purple')
        axes[1, 0].set_title('{level} after ripple'.format(level=level2))
    else:
        _plot_distribution(ds2, dims=DIMS, ax=axes[0, 0], color='purple')
        axes[0, 0].set_title('Baseline')
        _plot_distribution(ds2, dims=DIMS, ax=axes[1, 0], color='purple')
        axes[1, 0].set_title('Baseline')

    ds_change = (ds1 - ds2).mean(DIMS)
    ds_change.plot(
        x='time', y='frequency', ax=axes[0, 2], cmap='PRGn', center=0)
    axes[0, 2].set_title(
        '{covariate}: {level1} - {level2}'.format(
            level1=level1, level2=level2, covariate=covariate)
    )

    _plot_distribution(
        (ds1 - ds2).sel(time=0.0, method='backfill'), dims=DIMS,
        ax=axes[1, 2], color='midnightblue')
    axes[1, 2].set_title('Change after ripple')

    axes[0, 1].axvline(0, color='black', linestyle='--')
    axes[0, 2].axvline(0, color='black', linestyle='--')
    axes[1, 2].axhline(0, color='black', linestyle='--')

    plt.tight_layout()
    plt.suptitle(
        '{brain_area1}-{brain_area2}'.format(
            brain_area1=brain_area_pair[0],
            brain_area2=brain_area_pair[1]),
        fontsize=18, fontweight='bold')
    plt.subplots_adjust(top=0.90)


def _plot_distribution(
        ds, dims=None, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975],
        **plot_kwargs):
    alphas = np.array(quantiles)
    alphas[alphas > 0.5] = 1 - alphas[alphas > 0.5]
    alphas = (alphas / 0.5)
    alphas[alphas < 0.2] = 0.2

    for q, alpha in zip(quantiles, alphas):
        ds.quantile(q, dims).plot.line(alpha=alpha, **plot_kwargs)
