import logging


import argparse
from datetime import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import lrvsf


def get_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_gpus", type=int, default=-1)

    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, required=True)

    parser.add_argument(
        "--architecture", type=str, choices=["B32", "B16"], required=True
    )
    parser.add_argument("--conditioning", action="store_true")

    parser.add_argument("--run_name", type=str, default="debug")
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./saves")

    parser.add_argument(
        "--dataset_root", type=str, default=os.path.expanduser("~/DATA/LRVSF")
    )
    parser.add_argument("--n_samples", type=int, default=272_451)
    parser.add_argument("--shuffle", type=int, default=500)

    args = parser.parse_args()
    args = vars(args)

    if args["n_gpus"] == -1:
        args["n_gpus"] = torch.cuda.device_count()
    args["b_per_epoch"] = args["n_samples"] // args["batch_size"] // args["n_gpus"]
    args["run_name"] += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    args["run_folder"] = os.path.join(args["output_dir"], args["run_name"])

    return args


def create_exp_folder(args):
    if not os.path.exists(args["output_dir"]):
        os.mkdir(args["output_dir"])
    os.mkdir(args["run_folder"])


if __name__ == "__main__":
    args = get_cli_args()
    create_exp_folder(args)

    model = lrvsf.chkpt_utils.clip_init(
        args["architecture"],
        "models",
        len(lrvsf.constants.categories) if args["conditioning"] else None,
    )

    datasets = lrvsf.data.tardatasets.get_datasets(args)

    lrvsf.dist_utils.spawn_fn(lrvsf.run.run_fn, args["n_gpus"], model, datasets, args)
