import os
from argparse import ArgumentParser
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        help="path to folder containing all three datasets: train, valid, and test",
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        help="path to file containing bounding box annotations",
    )
    parser.add_argument(
        "--dataset_type",
        default="train",
        type=str,
        choices=["train", "val", "test", "all"],
        help="type of dataset i.e train, val, test, or 'all' to combine them",
    )

    parser.add_argument("--percentile", type=int, default=5, help="percentile")
    args = parser.parse_args()

    return args


def percentile_name(percentile):
    if percentile == 1:
        return "1st"
    elif percentile == 2:
        return "2nd"
    elif percentile == 3:
        return "3rd"
    else:
        return f"{percentile}th"


def slice_of_percentile(total_diagnoses, slice_sizes, slice_numbers, percentile):
    acc = 0
    for i, size in enumerate(slice_sizes):
        acc += size
        if acc >= total_diagnoses * percentile / 100:
            return slice_numbers[i]


def plot_diagnosis_distribution(
    dataset_path, annotations_path, percentile, dataset_type="train"
):
    if percentile < 0 or percentile >= 50:
        raise Exception("percentile invalid, must be between 0 and 49 inclusive")

    if not os.path.exists(annotations_path):
        raise Exception(
            f"The following annotations path does not exist: {annotations_path}"
        )

    if not os.path.exists(dataset_path):
        raise Exception(f"The following dataset path does not exist: {dataset_path}")

    dataset_type_filepath = f"{dataset_path}/singlecoil_{dataset_type}"
    if dataset_type != "all" and not os.path.exists(dataset_type_filepath):
        raise Exception(
            f"The following path to the specific dataset type does not exist: {dataset_type_filepath}"
        )

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
    df = pd.read_csv(annotations_path, dtype=dtype)
    df = df.dropna()

    if dataset_type != "all":
        filenames_in_dataset = [
            filename.split(".")[0] for filename in os.listdir(dataset_type_filepath)
        ]
        # Only use the rows whose filenames are present in the specified dataset
        df = df[df["file"].isin(filenames_in_dataset)]

    grouped_slices = df.groupby("slice")

    # Sort grouping by slice number
    grouped_slices = sorted(grouped_slices)
    slice_numbers = [slice_info[0] for slice_info in grouped_slices]
    slice_sizes = [len(slice_info[1]) for slice_info in grouped_slices]
    total_diagnoses = np.sum(slice_sizes)

    sop = partial(slice_of_percentile, total_diagnoses, slice_sizes, slice_numbers)
    fiftieth_percentile_slice = sop(50)
    left_percentile_slice = sop(percentile)
    right_percentile_slice = sop(100 - percentile)

    plt.bar(slice_numbers, slice_sizes)
    plt.axvline(
        fiftieth_percentile_slice,
        color="red",
        linestyle="--",
        label=f"{percentile_name(50)} percentile",
    )
    plt.axvline(
        left_percentile_slice,
        color="black",
        linestyle="--",
        label=f"{percentile_name(percentile)} percentile",
    )
    plt.axvline(
        right_percentile_slice,
        color="black",
        linestyle="--",
        label=f"{percentile_name(100 - percentile)} percentile",
    )
    plt.ylabel("Num Bounding Boxes")
    plt.xlabel("Slice")
    plt.title("Distribution of Bounding Boxes per Slices")
    plt.legend()
    plt.show()
    plt.clf()


if __name__ == "__main__":
    args = parse_args()
    plot_diagnosis_distribution(
        dataset_path=args.dataset_path,
        annotations_path=args.annotations_path,
        percentile=args.percentile,
        dataset_type=args.dataset_type,
    )
