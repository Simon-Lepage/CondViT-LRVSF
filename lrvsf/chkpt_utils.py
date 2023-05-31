from lrvsf.model import ConditionalViT, params
from lrvsf.dist_utils import unwrap

import torch

import os


def load_chkpt(save_path: str):
    print(f"Loading {save_path}")

    sd = torch.load(save_path, map_location="cpu")

    # We only experiment with B16/B32 so we can guess by the size of conv1
    architecture = f"B{sd['conv1.weight'].shape[-1]}"

    if "c_embedding.weight" in sd:
        n_categories = sd["c_embedding.weight"].shape[0]
    else:
        n_categories = None

    # Instanciante model based on guessed architecture and conditioning
    m = ConditionalViT(**params[architecture], n_categories=n_categories)
    m.load_state_dict(sd)

    return m.float().to("cuda"), "c_embedding.weight" in sd


def clip_init(archi, save_dir, n_categories=None):
    # CLIP initialization. In CLIP, logit_scale isn't part of the visual module
    model = ConditionalViT(**params[archi], n_categories=n_categories)
    sd = torch.load(f"{save_dir}/CLIP_{archi}_visual.pth", map_location="cpu")

    incompat = model.load_state_dict(sd, strict=False)

    if n_categories:
        # CondViT -> new cat. embeddings that weren't in CLIP.
        assert sorted(incompat.missing_keys) == [
            "c_embedding.weight",
            "c_pos_embedding",
            "logit_scale",
        ], incompat.missing_keys
        assert incompat.unexpected_keys == [], incompat.unexpected_keys
    else:
        # ViT
        assert incompat.missing_keys == ["logit_scale"], incompat.missing_keys
        assert incompat.unexpected_keys == [], incompat.unexpected_keys

    return model


def save_checkpoint(model_dir, model, fname):
    # Unwrap the model if it was DDP
    sd = unwrap(model).state_dict()

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    model_ckpt_path = os.path.join(model_dir, fname)
    torch.save(sd, model_ckpt_path)

    return model_ckpt_path
