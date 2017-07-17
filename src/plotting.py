import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.data_processing import read_netcdfs


def plot_power(path, group, brain_area, transform_func=None,
               figsize=(15, 5), vmin=0.5, vmax=2):
    ds = read_netcdfs(path, dim='session', group=group,
                      transform_func=transform_func).power
    mean_dims = ['session', 'tetrode']
    baseline = ds.isel(time=0).mean(mean_dims)
    raw_measure = ds.mean(mean_dims)
    diff_from_baseline = (ds / ds.isel(time=0)).mean(mean_dims)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    baseline.plot(ax=axes[0])
    axes[0].set_title('Baseline Power')
    raw_measure.plot(x='time', y='frequency', ax=axes[1])
    axes[1].set_title('Raw power')
    diff_from_baseline.plot(
        x='time', y='frequency', ax=axes[2],
        norm=LogNorm(vmin=vmin, vmax=vmax), cmap='RdBu_r',
        vmin=vmin, vmax=vmax, center=0)
    axes[2].set_title('Difference from baseline power')

    for ax in axes[1:3]:
        ax.axvline(0, color='black', linestyle='--')

    plt.tight_layout()
    plt.suptitle(brain_area, fontsize=18, fontweight='bold')
    plt.subplots_adjust(top=0.85)


def plot_connectivity(
    path, brain_area_pair, frequency, resolution, covariate,
    level1, level2=None, figsize=(15, 5),
        connectivity_measure='coherence_magnitude'):

    def get_data(level):
        group = '{resolution}/{covariate}/{level}/{measure}'.format(
            resolution=resolution, covariate=covariate, level=level,
            measure=connectivity_measure)

        def transform_func(ds):
            return ds.sel(
                tetrode1=ds.tetrode1[ds.brain_area1 == brain_area_pair[0]],
                tetrode2=ds.tetrode2[ds.brain_area2 == brain_area_pair[1]],
                frequency=frequency
            )
        return read_netcdfs(
            path, dim='session', group=group,
            transform_func=transform_func)[connectivity_measure]

    ds1 = get_data(level1)
    if level2 is not None:
        ds2 = get_data(level2)
    else:
        ds2 = ds1.isel(time=0)

    mean_dims = ['session', 'tetrode1', 'tetrode2']
    ds_level1 = ds1.mean(mean_dims)
    ds_level2 = ds2.mean(mean_dims)
    ds_change = (ds1 - ds2).mean(mean_dims)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if level2 is not None:
        ds_level1.plot(x='time', y='frequency', ax=axes[0], cmap='Purples')
        axes[0].set_title(level1)
    else:
        ds_level1.plot(ax=axes[0], color='purple')
        axes[0].set_title('Baseline')

    ds_level2.plot(x='time', y='frequency', ax=axes[1], cmap='Greens')
    axes[1].set_title(level2)

    ds_change.plot(
        x='time', y='frequency', ax=axes[2], cmap='PRGn', center=0)
    axes[2].set_title(
        '{covariate}: {level2} - {level1}'.format(
            level1=level1, level2=level2, covariate=covariate)
    )

    for ax in axes:
        ax.axvline(0, color='black', linestyle='--')

    plt.tight_layout()
    plt.suptitle(
        '{brain_area1} - {brain_area2}'.format(
            brain_area1=brain_area_pair[0],
            brain_area2=brain_area_pair[1]),
        fontsize=18, fontweight='bold')
    plt.subplots_adjust(top=0.85)
