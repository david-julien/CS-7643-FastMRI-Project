"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum

import torch
from torch.nn import functional as F

from fastmri import evaluate
from fastmri.models import Unet
from fastmri.pl_modules.mri_module import DistributedMetricSum

from .mri_module import MriModule


class Loss(Enum):
    MAE = "mae"
    WMAE = "wmae"


class UnetModule(MriModule):
    """
    Unet training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
        self,
        roi_bounding_boxes=None,
        loss=Loss.MAE.value,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        possible_loss_functions = [loss_type.value for loss_type in Loss]
        if loss not in possible_loss_functions:
            raise Exception(
                f"loss is set to: {loss} but must be one of {possible_loss_functions}"
            )

        self.ROI_SSIM = DistributedMetricSum()

        self.roi_bounding_boxes = roi_bounding_boxes
        self.loss = loss
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )

    def forward(self, image):
        return self.unet(image.unsqueeze(1)).squeeze(1)

    def weighted_mae(self, output, target, weights):
        return weights * torch.abs(output - target)

    def weighted_l1_loss(self, output, target, weights):
        _, Y, X = output.shape
        return torch.sum(self.weighted_mae(output, target, weights)) / (Y * X)

    def training_step(self, batch, batch_idx):
        output = self(batch.image)

        loss = None
        if self.loss == Loss.MAE.value:
            loss = F.l1_loss(output, batch.target)
            self.log("loss", loss)
        elif self.loss == Loss.WMAE.value:
            l1_loss = F.l1_loss(output, batch.target)
            loss = self.weighted_l1_loss(output, batch.target, batch.heatmap)
            self.log_dict({"loss": loss.detach(), "l1_loss": l1_loss.detach()})

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        results = {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "heatmap": batch.heatmap,
            "val_loss": F.l1_loss(output, batch.target),
        }

        if self.loss == Loss.WMAE.value:
            results.update(
                {
                    "val_loss": self.weighted_l1_loss(
                        output, batch.target, batch.heatmap
                    ),
                    "l1_loss": F.l1_loss(output, batch.target),
                }
            )

        return results

    def validation_step_end(self, val_logs):
        results = super().validation_step_end(val_logs)

        if self.loss == Loss.WMAE.value:
            # log weighted images to tensorboard
            if isinstance(val_logs["batch_idx"], int):
                batch_indices = [val_logs["batch_idx"]]
            else:
                batch_indices = val_logs["batch_idx"]
            for i, batch_idx in enumerate(batch_indices):
                if batch_idx in self.val_log_indices:
                    key = f"val_images_idx_{batch_idx}"
                    target = val_logs["target"][i].unsqueeze(0)
                    output = val_logs["output"][i].unsqueeze(0)
                    heatmap = val_logs["heatmap"][i].unsqueeze(0)
                    error = self.weighted_mae(output, target, heatmap)
                    output_focus_area = heatmap * output
                    output_focus_area = output_focus_area / output_focus_area.max()
                    target_focus_area = heatmap * target
                    target_focus_area = target_focus_area / target_focus_area.max()
                    error = error / error.max()
                    self.log_image(f"{key}/weighted_mae", error)
                    self.log_image(f"{key}/heatmap", heatmap)
                    self.log_image(
                        f"{key}/output_focus_area (heatmap * output)", output_focus_area
                    )
                    self.log_image(
                        f"{key}/target_focus_area (heatmap * target)", target_focus_area
                    )

        if self.roi_bounding_boxes is not None:
            # Aggregate roi ssim values
            roi_ssim_vals = defaultdict(dict)
            for i, fname in enumerate(val_logs["fname"]):
                slice_num = int(val_logs["slice_num"][i].cpu())
                maxval = val_logs["max_value"][i].cpu().numpy()
                output = val_logs["output"][i].cpu().numpy()
                target = val_logs["target"][i].cpu().numpy()

                min_x, min_y, width, height = self.roi_bounding_boxes[slice_num]
                target_roi = target[min_y : min_y + height, min_x : min_x + width]
                output_roi = output[min_y : min_y + height, min_x : min_x + width]
                roi_ssim_vals[fname][slice_num] = torch.tensor(
                    evaluate.ssim(
                        target_roi[None, ...], output_roi[None, ...], maxval=maxval
                    )
                ).view(1)

            results.update({"roi_ssim_vals": roi_ssim_vals})

        return results

    def validation_epoch_end(self, val_logs):
        results = super().validation_epoch_end(val_logs)

        if self.roi_bounding_boxes is not None:
            # Log ROI SSIM values
            roi_ssim_vals = defaultdict(dict)
            mse_vals = defaultdict(dict)
            # use dict updates to handle duplicate slices
            for val_log in val_logs:
                for k in val_log["mse_vals"].keys():
                    mse_vals[k].update(val_log["mse_vals"][k])
                for k in val_log["roi_ssim_vals"].keys():
                    roi_ssim_vals[k].update(val_log["roi_ssim_vals"][k])

            # check to make sure we have all files in all metrics
            assert mse_vals.keys() == roi_ssim_vals.keys()

            # apply means across image volumes
            metrics = {"roi_ssim": 0}
            local_examples = 0
            for fname in mse_vals.keys():
                local_examples = local_examples + 1
                metrics["roi_ssim"] = metrics["roi_ssim"] + torch.mean(
                    torch.cat([v.view(-1) for _, v in roi_ssim_vals[fname].items()])
                )

            # reduce across ddp via sum
            metrics["roi_ssim"] = self.ROI_SSIM(metrics["roi_ssim"])
            tot_examples = self.TotExamples(torch.tensor(local_examples))

            self.log("val_metrics/roi_ssim", metrics["roi_ssim"] / tot_examples)

        return results

    def test_step(self, batch, batch_idx):
        output = self.forward(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
