#!/usr/bin/env python
import sys
sys.path.append(".")
import logging
import click
import re
import io
from functools import partial
import pandas as pd
import numpy as np
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def calc_basic_train_test_stats(df):
    g = df.groupby('is_train')
    res = pd.concat([
        g.size().rename("# of series"),
        g['object_id'].nunique().rename("# of obj"),
        g['timestamp_min'].min().rename("min time"),
        g['timestamp_max'].max().rename("max time"),
    ], axis=1)
    for c in ['min time', 'max time']:
        res[c] = res[c].values.astype('<M8[m]')  # trunc to minutes
        res[c] = res[c].dt.strftime("%Y-%m-%d %H:%M")

    res = res.reindex([True, False])
    res = res.rename({True: "train", False: "test"})
    res = res.transpose()
    return res


def plot_series_by_object_id_and_time(df, color='max_phase_num', reverse_legend=False, filename=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    columns = ['object_id', 'pipeline', 'timestamp_min', 'max_phase_num', 'is_train']
    if color not in columns:
        columns.append(color)
    df = df[columns].copy()
    df['x'] = df['timestamp_min'].rank(method='dense')
    df['y'] = df['object_id'].rank(method='dense')
    if color=='max_phase_num':
        pal = sns.color_palette('Set1', 4) + [sns.xkcd_rgb['grey'], sns.xkcd_rgb['grey']]
    else:
        pal = sns.color_palette('Set1', df[color].nunique())
    ranks = df[color].rank(method='dense').fillna(0).astype(int)
    rank2color = {x: pal[int(x)-1] if x >= 1 else sns.xkcd_rgb['grey'] for x in ranks.unique()}
    rank2label = df.groupby(ranks)[color].first()

    c = ranks.map(rank2color)
    ax.scatter(
        x=df['x'],
        y=df['y'], 
        marker=',', lw=0, s=10,
        c=c
    )

    ylabels = df.groupby('y')['object_id'].first()
    ax.set_ylabel("object id")
    ax.set_yticklabels([ylabels.get(y) for y in ax.get_yticks()])

    ylabels = df.groupby('y')['object_id'].first()
    ax.set_ylabel("object id")
    ax.set_yticklabels([ylabels.get(y) for y in ax.get_yticks()])

    xlabels = df.groupby('x')['timestamp_min'].first().dt.strftime("%Y-%m-%d")
    ax.set_xlabel("time")
    ax.set_xticklabels([xlabels.get(x) for x in ax.get_xticks()])

    handles = []
    for r, label in sorted(rank2label.items()):
        c = rank2color.get(r)
        if c is None:
            logging.error("missing color for rank %d", r)
            continue
        handles.append(mpatches.Patch(color=c, label=str(r) + " " + str(label)))
    if reverse_legend:
        handles = list(reversed(handles))
    ax.legend(handles=handles, 
        title={
            'max_phase_num': "max phase num",
            'pipeline': 'pipeline num'
        }.get(color, color), 
        loc='center left', bbox_to_anchor=(1, 0.5))

    sns.despine(ax=ax)
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        plt.show()

def plot_target_heatmap_by_cleaning_num(df, filename=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    g = df.groupby(['object_id', 'f_total_cleaning_num'])['target'].first().unstack(level=1)
    sns.heatmap(g, ax=ax, vmin=0, vmax=2*1e6)
    g2 = df[~df.is_train].groupby(['object_id', 'f_total_cleaning_num'])['target'].first().fillna(1).unstack(level=1)\
        .reindex(g.index)
    sns.heatmap(g2, ax=ax, cmap='Greens', cbar=False, vmin=0, vmax=2)
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        plt.show()

def show_values_for_single_process(df, object_id, column='target', filename=None, figsize=(15, 8)):
    g = df[df.object_id==object_id].sort_values(by='f_total_cleaning_num')
    fig, ax = plt.subplots(figsize=figsize)
    g.plot(kind='line', x='f_total_cleaning_num', y=column, ax=ax, legend=None)
    ax.set_ylim(min(0, ax.get_ylim()[0]), None)
    ax.set_ylabel(column)
    ax.yaxis.grid(True, alpha=0.4)
    sns.despine(ax=ax)
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        plt.show()


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True), default="data/processed/train_test.hdf5")
def main(input_filepath):
    logging.info("reading %s", input_filepath)
    train_test = pd.read_hdf(input_filepath, "train_test")
    logging.info(f"shape: {train_test.shape}")

    basic_train_test_stats = calc_basic_train_test_stats(train_test)
    basic_train_test_stats.to_latex("figures/_basic_train_test_stats.latex", index_names=False)

    fn = "figures/fig_series_by_object_id_and_time.png"
    logging.info(f"generating {fn}")
    plot_series_by_object_id_and_time(train_test, filename=fn)

    fn = "figures/fig_target_heatmap_by_cleaning_num.png"
    logging.info(f"generating {fn}")
    plot_target_heatmap_by_cleaning_num(train_test, filename=fn)

    fn = "figures/fig_targets_for_object_id_405.png"
    logging.info(f"generating {fn}")
    show_values_for_single_process(train_test, object_id=405, column='target', filename=fn)

    fn = "figures/fig_max_pressure_for_object_id_405.png"
    logging.info(f"generating {fn}")
    show_values_for_single_process(train_test, object_id=405, 
        column='f_phase_1_max_supply_pressure', filename=fn,
        figsize=(15, 5)
    )

if __name__ == "__main__":
    sns.set(style="white")

    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_fmt)
    main()
