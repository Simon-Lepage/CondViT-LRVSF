from lrvsf.dist_utils import unwrap
from lrvsf.train.loss import model_inference, compute_loss

import torch
import torch.distributed as dist

import faiss
import numpy as np

from tqdm import tqdm
import time
import logging

logger = logging.getLogger(__name__)


def train(
    model,
    optimizer,
    scheduler,
    scaler,
    trainloader,
    tb_writer,
    b_per_epoch,
    epoch,
    device,
):
    model.train()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    n_done_steps = b_per_epoch * epoch

    freq_log = 5
    batch_time_acc = 0

    end = time.perf_counter()
    for i, batch in enumerate(trainloader):
        current_step = n_done_steps + i

        optimizer.zero_grad(set_to_none=True)
        batch = [b.to(device, non_blocking=True) for b in batch]

        bs = batch[0].shape[0]
        data_time = time.perf_counter() - end
        with torch.cuda.amp.autocast():
            e_simple, e_complex = model_inference(model, *batch)
            loss, metrics = compute_loss(
                e_simple, e_complex, unwrap(model).logit_scale, device, rank
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        logit_scale = unwrap(model).logit_scale
        with torch.no_grad():
            logit_scale.clamp_(0, 4.6052)

        batch_time = time.perf_counter() - end
        batch_time_acc += batch_time

        if i % freq_log == 0:
            log_data = {
                "step": current_step,
                "loss": loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale": model.module.logit_scale.data.item(),
                "lr": optimizer.param_groups[-1]["lr"],
                "samples_per_second": world_size * bs * freq_log / batch_time_acc,
            }
            log_data.update(metrics)

            logger.info(
                f"[{rank}] Train Epoch : {epoch} "
                f"[{i}/{b_per_epoch}]"
                f"[{(100.0 * i / b_per_epoch):.0f}%]"
                f"\tLoss: {loss.item():.6f}"
                f"\tAccuracy: {(metrics['batch_acc']*100.0):.2f}"
                f"\tData (t) {data_time:.3f}"
                f"\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[-1]['lr']:5f}"
                f"\tLogit_scale {model.module.logit_scale.data:.3f}"
            )

            if rank == 0:
                for name, val in log_data.items():
                    name = "train/" + name
                    tb_writer.add_scalar(name, val, current_step)

            batch_time_acc = 0

        end = time.perf_counter()


@torch.no_grad()
def valid(model, validloader, distloader, tb_writer, epoch, device):
    logger.info("VALIDATION.")
    m = unwrap(model)
    m.eval()

    # Embed distractors
    dist_embs = []
    for (imgs,) in tqdm(distloader):
        imgs = imgs.to(device)
        embs = torch.nn.functional.normalize(m(imgs))
        dist_embs.append(embs.cpu())
    dist_embs = torch.cat(dist_embs, axis=0)

    # Paired query/galery embeddings
    g_embs = []
    q_embs = []
    for b in validloader:
        b = [elt.to(device) for elt in b]
        e_simple, e_complex = model_inference(m, *b)
        g_embs.append(e_simple.cpu())
        q_embs.append(e_complex.cpu())

    g_embs = torch.cat(g_embs, axis=0)
    q_embs = torch.cat(q_embs, axis=0)
    print(q_embs.shape, g_embs.shape)

    # Build faiss index
    index = faiss.IndexFlatIP(g_embs.shape[1])
    index.add(g_embs.numpy())
    index.add(dist_embs.numpy())
    res = index.search(q_embs.numpy(), 100)

    # Compute metrics
    target = np.arange(q_embs.shape[0])
    recalls = np.cumsum(res[1] == target[:, None], axis=1).mean(axis=0)
    metrics = {f"val/R@{i+1}": recalls[i] for i in [0, 4, 9, 19, 49, 99]}

    target = torch.tensor(target).to(device)
    logits = m.logit_scale.exp() * q_embs.to(device) @ g_embs.to(device).t()
    loss = (
        torch.nn.functional.cross_entropy(logits, target)
        + torch.nn.functional.cross_entropy(logits.t(), target)
    ) / 2

    metrics["val/loss"] = loss.cpu().item()

    for k in metrics:
        tb_writer.add_scalar(k, metrics[k], epoch)

    print(metrics)
    return metrics
