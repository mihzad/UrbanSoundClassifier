import os
from pathlib import Path
import matplotlib.pyplot as plt
from audio_loading_utils import UrbanSoundDataset, TransformSubset
import numpy as np

def classcount_hist(class_counts, class_labels):

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_labels, class_counts)

    # display exact counts per hist batch
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            str(height),
            ha="center", va="bottom"
        )

    plt.xticks(rotation=90)
    plt.xlabel("Клас")
    plt.ylabel("Кількість wav-ів")
    plt.title("Розподіл аудіозразків по класах")
    plt.tight_layout()
    plt.show()

def duration_hist(set: UrbanSoundDataset | TransformSubset):
    if isinstance(set, TransformSubset):
        meta = set.dataset.meta.iloc[set.indices].copy()
    else:
        meta = set.meta
    meta['duration'] = meta['end'] - meta['start']

    # Average duration
    avg = meta['duration'].mean()
    print("Average duration (s):", avg)

    # Histogram with 0.2s bins up to 4s excluding it
    bins = np.arange(0, 4.0+0.2, 0.2)
    hist, edges = np.histogram(meta['duration'], bins=bins)
    count_4s = (meta['duration'] == 4.0).sum()
    hist[-1] -= count_4s #reserve 4s vals for special bar

    plt.figure(figsize=(10, 6))
    bars = plt.bar(edges[:-1], hist, width=0.18, align='edge', edgecolor='black')

    #special bar for 4s
    count_4s = (meta['duration'] == 4.0).sum()
    if count_4s > 0:
        bar_4s = (plt.bar(4.0, count_4s, width=0.18, align='edge',
                    edgecolor='black', color='green', label='4s clips'))[0]
        plt.text(
            bar_4s.get_x() + bar_4s.get_width() / 2,  # x position
            bar_4s.get_height(),  # y position
            str(count_4s),  # text = count
            ha='center', va='bottom', fontsize=8
        )

    # Add exact counts above each bin
    for bar, count in zip(bars, hist):
        if count > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # x position
                bar.get_height(),  # y position
                str(count),  # text = count
                ha='center', va='bottom', fontsize=8
            )

    plt.xlabel("Тривалість (с)")
    plt.ylabel("К-ть wav-ів")
    plt.title(f"Тривалості wav-ів для UrbanSound8K (avg={avg:.2f}с)")
    plt.xticks(edges, rotation=45)
    plt.show()