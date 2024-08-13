import os

import torch
from torch.utils.tensorboard import SummaryWriter

from .dist_utils import setup, cleanup
from . import data
from .train import loops
from .chkpt_utils import save_checkpoint

import logging

logger = logging.getLogger(__name__)


def logging_config(rank, args):
    logging.basicConfig(
        filename=os.path.join(args["run_folder"], f"log_{rank}.log"),
        level=logging.DEBUG,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)

    logging.getLogger().addHandler(console)
    return logging.getLogger(__name__)


def run_fn(rank, world_size, model, datasets, args):
    setup(rank, world_size)

    # Needed for split_by_node and split_by_worker 
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.manual_seed(rank)

    logger = logging_config(rank, args)
    logger.info(f"Starting run {args['run_name']}")
    logger.info(args)

    # Prepare training
    device = torch.device("cuda", rank)
    model = torch.nn.parallel.DistributedDataParallel(model.to(rank), device_ids=[rank])
    trainloader, validloader, distloader = data.tardatasets.make_loaders(
        *datasets, args
    )

    tb_writer = SummaryWriter(f"{args['run_folder']}")

    total_steps = args["b_per_epoch"] * args["n_epochs"]
    logger.info(
        f"N epochs : {args['n_epochs']}"
        f" - Train step per epoch : {args['b_per_epoch']}"
        f" - Total steps {total_steps}"
    )

    # Prepare warm-up
    ## Parameter selection
    warm_up_params = [model.module.proj]
    if args["conditioning"]:
        warm_up_params.extend(
            [model.module.c_embedding.weight, model.module.c_pos_embedding]
        )
    ## Optimizer / Scheduler
    optimizer = torch.optim.AdamW(warm_up_params, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.02, total_iters=args["b_per_epoch"]
    )
    scaler = torch.cuda.amp.GradScaler()

    best_r1 = 0
    best_r1_epoch = 0

    for epoch in range(args["n_epochs"]):
        os.environ["TRAINING_EPOCH"] = str(epoch)

        # End of warm-up
        if epoch == 1:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(args["n_epochs"] - 1) * args["b_per_epoch"],
                eta_min=1e-8,
            )

        try:
            logger.info(f"EPOCH {epoch}")
            loops.train(
                model,
                optimizer,
                scheduler,
                scaler,
                trainloader,
                tb_writer,
                args["b_per_epoch"],
                epoch,
                device,
            )

            torch.distributed.barrier()
        except StopIteration:
            continue

        if rank == 0 and epoch % args["save_freq"] == 0:
            metrics = loops.valid(
                model, validloader, distloader, tb_writer, epoch, device
            )

            save_checkpoint(args["run_folder"], model, "last_model.pth")
            if metrics["val/R@1"] >= best_r1:
                best_r1 = metrics["val/R@1"]
                best_r1_epoch = epoch
                save_checkpoint(args["run_folder"], model, "best_validation_model.pth")

        torch.distributed.barrier()

    if rank == 0:
        logger.info(f"Best epoch : {best_r1_epoch} : {best_r1}%.")

    cleanup()
