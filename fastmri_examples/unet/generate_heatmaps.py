import os
from argparse import ArgumentParser

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

NUM_SLICES = 50
MAP_DIM = 320


def min_max_scaler(heatmaps, min):
    """
    Normalizes the heatmaps to be between <min> and 1.
    Source: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

    Args:
        heatmaps: N x M x M ndarray, containing N heatmaps, one for each slice
        min: float value which designates the minimum value in the heatmap after normalization

    Returns: N x M x M ndarray, containing N normalized heatmaps, one for each slice

    """
    numerator = heatmaps - heatmaps.min(axis=(1, 2), keepdims=True)
    denominator = heatmaps.max(axis=(1, 2), keepdims=True) - heatmaps.min(
        axis=(1, 2), keepdims=True
    )
    # Avoid dividing by 0
    denominator[denominator == 0] = 1
    heatmaps = numerator / denominator

    # Scale values to be between min value and 1 inclusive
    heatmaps = heatmaps * (1 - min) + min

    return heatmaps


# calculate heatmap
def generate_heatmaps(
    dataset_path,
    annotations_path,
    dataset_type="train",
    min_heatmap_value=0.2,
):
    if min_heatmap_value < 0 or min_heatmap_value > 1:
        raise Exception(
            f"min heatmap value must be between 0 and 1 inclusive, however it is currently set to {min_heatmap_value}"
        )

    if dataset_type not in ["train", "val", "test", "all"]:
        raise Exception(
            f"dataset type must be one of train, val, or test, however it is currently set to {dataset_type}"
        )

    if not os.path.exists(annotations_path):
        raise Exception(
            f"The following annotations path does not exist: {annotations_path}"
        )

    if not os.path.exists(dataset_path):
        raise Exception(f"The following dataset path does not exist: {dataset_path}")

    filenames = list()
    dataset_type_filepath = f"{dataset_path}/singlecoil_{dataset_type}"
    if dataset_type != "all":
        if not os.path.exists(dataset_type_filepath):
            raise Exception(
                f"The following path to the specific dataset type does not exist: {dataset_type_filepath}"
            )
        filenames = os.listdir(dataset_type_filepath)

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

    print("Generating heatmaps...")
    heatmaps = np.zeros((NUM_SLICES, MAP_DIM, MAP_DIM))
    num_heatmaps = 0
    for index, row in tqdm(annotations_df.iterrows(), total=len(annotations_df)):
        if dataset_type != "all" and not any(row.file in file for file in filenames):
            # Filter out any files that are not in the specified dataset
            continue

        heatmaps[row.slice, row.y : row.y + row.height, row.x : row.x + row.width] += 1
        num_heatmaps += 1

    heatmaps = min_max_scaler(heatmaps, min=min_heatmap_value)

    # Flip the heatmaps across the y-axis, since the bounding boxes are upside down
    heatmaps[:] = heatmaps[:, ::-1, :]

    print(f"{num_heatmaps} heatmaps generated for {dataset_type} set")

    return heatmaps


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        help="path to folder containing all three datasets: train, valid, and test",
        required=True,
    )
    parser.add_argument(
        "--annotations_path",
        help="path to file containing bounding box annotations",
        required=True,
    )
    parser.add_argument(
        "--dataset_type",
        default="train",
        choices=["train", "test", "val", "all"],
        help="type of dataset to get the heatmap from. i.e train, val, test, or all",
    )
    parser.add_argument(
        "--min_heatmap_value",
        type=float,
        default=0.2,
        help="when the heatmap is normalized, all values in it will be between min_heatmap_value and 1 inclusive",
    )
    parser.add_argument(
        "--bounding_box_min",
        type=float,
        help="when specified, a bounding box is plotted over the heatmap where the edges of the heatmap are greater "
        "than or equal to this min value",
    )
    args = parser.parse_args()

    return args


def generate_bounding_box(heatmap, min=0.2):
    """

    Args:
        heatmap: M x M ndarray representing heatmap
        min: float where the edges of the bounding box are greater than this value

    Returns: tuple of size 4 including min_x, min_y, width, and height

    """
    nonzero_indexes = np.nonzero(heatmap > min)
    if len(nonzero_indexes[0]) < 1:
        return 0, 0, 0, 0

    min_x = nonzero_indexes[1].min()
    max_x = nonzero_indexes[1].max()
    min_y = nonzero_indexes[0].min()
    max_y = nonzero_indexes[0].max()
    return min_x, min_y, max_x - min_x, max_y - min_y


def generate_bounding_boxes(heatmaps):
    """

    Args:
        heatmaps: N x M x M ndarray, containing N heatmaps, one for each slice

    Returns: N X 4 ndarray, containing the bounding box coordinates for each slice's heatmap

    """
    bounding_boxes = np.zeros((NUM_SLICES, 4))
    for slc in range(NUM_SLICES):
        bounding_boxes[slc] = generate_bounding_box(heatmaps[slc])
    return bounding_boxes


def plot():
    args = parse_args()
    heatmaps = generate_heatmaps(
        dataset_path=args.dataset_path,
        annotations_path=args.annotations_path,
        dataset_type=args.dataset_type,
        min_heatmap_value=args.min_heatmap_value,
    )

    fig, ax = plt.subplots(10, 5, figsize=(20, 20))
    ax = ax.flatten()
    for slc in range(NUM_SLICES):
        ax[slc].set_title(slc)
        heatmap = ax[slc].imshow(heatmaps[slc], cmap="hot", vmin=0)

        if args.bounding_box_min is not None:
            min_x, min_y, width, height = generate_bounding_box(
                heatmaps[slc], min=args.bounding_box_min
            )
            # Source: https://stackoverflow.com/questions/37435369/how-to-draw-a-rectangle-on-image
            rect = patches.Rectangle(
                (min_x, min_y),
                width,
                height,
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
            )

            ax[slc].add_patch(rect)
        fig.colorbar(heatmap, ax=ax[slc])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot()
