import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

NUM_SLICES = 50
MAP_DIM = 320


# calculate heatmap
def generate_heatmaps(dataset_path, annotations_path, dataset_type="train"):
    training_filenames = os.listdir(f"{dataset_path}/singlecoil_{dataset_type}")
    dtype = {
        "file": str,
        "slice": "Int64",
        "study_level": str,
        "x": "Int64",
        "y": "Int64",
        "width": "Int64",
        "height": "Int64",
        "label": str,
    }

    annotations_df = pd.read_csv(annotations_path, dtype=dtype)
    annotations_df = annotations_df.dropna()

    heatmaps = np.zeros((NUM_SLICES, MAP_DIM, MAP_DIM))
    for index, row in annotations_df.iterrows():
        filename = row.file + ".h5"
        if filename not in training_filenames:
            continue

        heatmaps[row.slice, row.y : row.y + row.height, row.x : row.x + row.width] += 1

    # Normalize the data using min/max scaling
    numerator = heatmaps - heatmaps.min(axis=(1, 2), keepdims=True)
    denominator = heatmaps.max(axis=(1, 2), keepdims=True) - heatmaps.min(
        axis=(1, 2), keepdims=True
    )
    # Avoid dividing by 0
    denominator[denominator == 0] = 1
    heatmaps = numerator / denominator

    min_val = 0.2
    heatmaps = (1 - min_val) * heatmaps + min_val

    # Flip the heatmaps across the y-axis, since the bounding boxes are upside down
    heatmaps[:] = heatmaps[:, ::-1, :]

    return heatmaps


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        help="path to folder containing all three datasets: train, valid, and test",
    )
    parser.add_argument(
        "--annotations_path", help="path to file containing bounding box annotations"
    )
    parser.add_argument(
        "--dataset_type",
        default="train",
        help="type of dataset to get the heatmap from. i.e train, val, or test",
    )
    args = parser.parse_args()

    if args.dataset_type not in ['train', 'val', 'test']:
        raise Exception("--dataset_type must be one of train, val, or test")

    return args


if __name__ == "__main__":
    args = parse_args()
    heatmaps = generate_heatmaps(
        dataset_path=args.dataset_path,
        annotations_path=args.annotations_path,
        dataset_type=args.dataset_type,
    )

    fig, ax = plt.subplots(10, 5, figsize=(20, 20))
    ax = ax.flatten()
    for slc in range(NUM_SLICES):
        ax[slc].set_title(slc)
        heatmap = ax[slc].imshow(heatmaps[slc], cmap="hot", vmin=0)
        fig.colorbar(heatmap, ax=ax[slc])

    plt.tight_layout()
    plt.show()