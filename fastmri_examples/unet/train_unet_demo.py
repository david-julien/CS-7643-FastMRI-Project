"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from generate_heatmaps import generate_heatmaps, generate_rois
from pytorch_lightning.loggers import TensorBoardLogger

from fastmri.data import FastMRIRawDataSample
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule, UnetModule
from fastmri.pl_modules.unet_module import Loss


class MyProgressBar(pl.callbacks.TQDMProgressBar):
    # This class prevents the progress bar from printing on a new line every time.
    # Source: https://github.com/Lightning-AI/pytorch-lightning/issues/15283#issuecomment-1289654353

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.position = 0
        bar.leave = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.disable = True
        return bar


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )

    if args.loss == Loss.WMAE.value:
        print("Using weighted mean average error")
    else:
        print("Using mean average error")

    train_heatmaps = generate_heatmaps(
        dataset_path=args.data_path,
        annotations_path=args.annotations_path,
        dataset_type="train",
        heatmap_min_value=args.heatmap_min_value,
    )

    bb_heatmaps = generate_heatmaps(
        dataset_path=args.data_path,
        annotations_path=args.annotations_path,
        dataset_type="train",
        heatmap_min_value=0,
    )

    roi_bounding_boxes = generate_rois(bb_heatmaps, args.roi_min_value)

    # use random masks for train transform, fixed masks for val transform
    train_transform = UnetDataTransform(
        args.challenge, mask_func=mask, use_seed=False, heatmaps=train_heatmaps
    )
    val_transform = UnetDataTransform(
        args.challenge, mask_func=mask, heatmaps=train_heatmaps
    )

    test_transform = UnetDataTransform(args.challenge)

    print('prune_left_bound_idx:', args.prune_left_bound_idx, 'prune_right_bound_idx:', args.prune_right_bound_idx)
    def custom_train_filter(raw_sample: FastMRIRawDataSample) -> bool:
        if raw_sample.slice_ind < args.prune_left_bound_idx or raw_sample.slice_ind > args.prune_right_bound_idx:
            return False
        return True


    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        # customly added filter to prune edge slices
        train_filter=custom_train_filter,
        val_filter=custom_train_filter,
        test_filter=custom_train_filter,
    )

    # ------------
    # model
    # ------------
    model = UnetModule(
        roi_bounding_boxes=roi_bounding_boxes,
        loss=args.loss,
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    num_gpus = 2
    backend = "ddp"
    batch_size = 1 if backend == "ddp" else num_gpus

    # set defaults based on optional directory config
    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "unet" / "unet_demo"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        help="Path to annotations file",
    )
    parser.add_argument(
        "--roi_min_value",
        type=float,
        default=0.2,
        help="All values outside the ROI bounding box are less than or equal to the roi_min_value",
    )
    parser.add_argument(
        "--heatmap_min_value",
        type=float,
        default=0.2,
        help="This is the minimium value that any given cell in the heatmap will take after normalization",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=[Loss.MAE.value, Loss.WMAE.value],
        default=Loss.MAE.value,
        help="Type of loss function to be used. wmae is the weighted mae. If you specify wmae"
        " you must specify the --annotations_path",
    )
    parser.add_argument(
        "--prune_left_bound_idx",
        default=12,
        type=int,
        help="Prune all slices with idx less than the left bound",
    )
    parser.add_argument(
        "--prune_right_bound_idx",
        default=29,
        type=int,
        help="Prune all slices with idx greater than the right bound",
    )

    # data config with path to fastMRI data and batch size
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(data_path=data_path, batch_size=batch_size, test_path=None)

    # module config
    parser = UnetModule.add_model_specific_args(parser)
    parser.set_defaults(
        in_chans=1,  # number of input channels to U-Net
        out_chans=1,  # number of output chanenls to U-Net
        chans=32,  # number of top-level U-Net channels
        num_pool_layers=4,  # number of U-Net pooling layers
        drop_prob=0.0,  # dropout probability
        lr=0.001,  # RMSProp learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight decay regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        strategy=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
        logger=TensorBoardLogger(
            str(default_root_dir), name="lightning_logs", flush_secs=60
        ),
    )

    args = parser.parse_args()

    if len(args.annotations_path) == 0:
        raise Exception("must specify annotations path to calculate ROI SSIM")

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        ),
        MyProgressBar(),
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
