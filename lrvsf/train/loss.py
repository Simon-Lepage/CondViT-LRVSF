import torch
import torch.nn.functional as F

from lrvsf.train.gatherlayer import GatherLayer

import logging

logger = logging.getLogger(__name__)


@torch.cuda.amp.autocast()
def model_inference(model, simple_img, complex_img, conditioning=None):
    if conditioning is not None:
        e_simple = model(simple_img)  # Simple image encoded without conditioning
        e_complex = model(complex_img, conditioning)
    else:
        e_simple = model(simple_img)
        e_complex = model(complex_img)

    e_simple = F.normalize(e_simple)
    e_complex = F.normalize(e_complex)

    return e_simple, e_complex


@torch.cuda.amp.autocast()
def compute_loss(e_simple, e_complex, logit_scale, device, rank):
    # Sharded loss : gather embeddings
    all_simple = torch.cat(GatherLayer.apply(e_simple), dim=0)
    all_complex = torch.cat(GatherLayer.apply(e_complex), dim=0)

    # Scaled cosine similarity
    logits_per_complex = logit_scale.exp() * e_complex @ all_simple.t()
    logits_per_simple = logit_scale.exp() * e_simple @ all_complex.t()

    ground_truth = torch.arange(logits_per_complex.shape[0], device=device)
    ground_truth += rank * logits_per_complex.shape[0]

    total_loss = (
        F.cross_entropy(logits_per_simple, ground_truth)
        + F.cross_entropy(logits_per_complex, ground_truth)
    ) / 2

    #  Compute additional batch metrics
    metrics = {}
    with torch.no_grad():
        metrics["batch_acc"] = (
            (torch.argmax(logits_per_complex, dim=1) == ground_truth)
            .float()
            .mean()
            .item()
        )
        metrics["align"] = (
            (e_simple - e_complex).norm(dim=1).pow(2).mean().item()
        )  # Computed only locally.

    return total_loss, metrics
