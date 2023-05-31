import os
from . import transforms as tf
from lrvsf.constants import categories
import random
import webdataset as wds
import json
from PIL import Image
from io import BytesIO
from functools import partial


# Functions used outside
def get_datasets(args):
    train_tfs = tf.train_tf((224, 224))
    valid_tfs = tf.valid_tf((224, 224))

    train_tars = get_tars(args["dataset_root"], "TRAIN")
    valid_tars = get_tars(args["dataset_root"], "VALID", lambda p: "prods.tar" in p)
    dist_tars = get_tars(args["dataset_root"], "VALID", lambda p: "dist" in p)

    trainset = create_webdataset(train_tars, train_tfs, args, "train")
    validset = create_webdataset(valid_tars, valid_tfs, args, "valid")
    distset = create_dist_dataset(dist_tars, valid_tfs)

    return trainset, validset, distset


def make_loaders(trainset, validset, distset, args):
    trainloader = (
        wds.WebLoader(
            trainset,
            batch_size=args["batch_size"],
            num_workers=8,
            drop_last=False,
            persistent_workers=False,
        )
        .unbatched()
        .shuffle(args["shuffle"])
        .batched(args["batch_size"], partial=False)
        .with_epoch(nsamples=args["b_per_epoch"])
    )

    validloader = wds.WebLoader(
        validset,
        batch_size=256,
        num_workers=1, # Single worker because single tarfile.
        shuffle=False,
        drop_last=False,
    )

    distloader = wds.WebLoader(
        distset, batch_size=256, num_workers=15, shuffle=False, drop_last=False
    )

    return trainloader, validloader, distloader


# Inner classes & tools
# Deterministic epoch-wise shuffling.
class epoch_detshuffle(wds.PipelineStage):
    def __init__(self, bufsize=1000, initial=100, seed=0):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed

    def run(self, src):
        epoch = int(os.environ["TRAINING_EPOCH"])
        print(f"Reading epoch {epoch}")
        rng = random.Random()
        rng.seed(self.seed + epoch)
        return wds.filters._shuffle(src, self.bufsize, self.initial, rng)


# Get tarfiles from folder
def get_tars(root, folder, filter_fn=lambda p: True):
    root = os.path.join(root, folder)
    files = [os.path.join(root, f) for f in os.listdir(root) if filter_fn(f)]
    return files


def reverse_type(d):
    acc = {"SIMPLE": [], "COMPLEX": [], "PARTIAL_COMPLEX": []}
    for k, v in d.items():
        acc[v["TYPE"]].append(k)
    return acc


def preprocess_dataset(
    row, image_key, metadata_key, image_transform, conditioning=None
):
    output = {}

    d = json.loads(row[metadata_key])
    type_2_keys = reverse_type(d)

    selected_simple_img = random.choice(type_2_keys["SIMPLE"])
    selected_complex_img = random.choice(type_2_keys["COMPLEX"] + type_2_keys["PARTIAL_COMPLEX"])

    output["simple"] = Image.open(
        BytesIO(row[f"{selected_simple_img}.{image_key}"])
    ).convert("RGB")
    output["complex"] = Image.open(
        BytesIO(row[f"{selected_complex_img}.{image_key}"])
    ).convert("RGB")

    output["simple"] = image_transform(output["simple"])
    output["complex"] = image_transform(output["complex"])

    if conditioning:
        output["conditioning"] = categories.index(d[selected_simple_img]["CATEGORY"])

    return output


def create_webdataset(
    urls,
    image_transform,
    args,
    mode,
    image_key="jpg",
    metadata_key="json",
    handler=wds.handlers.warn_and_continue,
):
    transform_fn = partial(
        preprocess_dataset,
        image_key=image_key,
        metadata_key=metadata_key,
        image_transform=image_transform,
        conditioning=args["conditioning"],
    )

    if args["conditioning"]:
        t = ("simple", "complex", "conditioning")
    else:
        t = ("simple", "complex")

    if mode == "train":
        ds = wds.DataPipeline(
            wds.SimpleShardList(urls),
            epoch_detshuffle(),
            wds.split_by_node,  # Split by GPU
            wds.split_by_worker,  # Split by DataLoader worker
            wds.tarfile_to_samples(),
            wds.map(transform_fn, handler=handler),
            wds.to_tuple(*t),
        )
    elif mode == "valid":
        ds = wds.DataPipeline(
            wds.SimpleShardList(urls),
            # Single GPU, no need to shuffle / split by node.
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.map(transform_fn, handler=handler),
            wds.to_tuple(*t),
        )
    else:
        raise ValueError

    return ds


def process_dist(row, tf):
    img = Image.open(BytesIO(row["jpg"])).convert("RGB")
    out = {"image": tf(img)}
    return out


def create_dist_dataset(urls, valid_tf):
    partial_fn = partial(process_dist, tf=valid_tf)

    ds = wds.DataPipeline(
        wds.SimpleShardList(urls),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.map(partial_fn, handler=wds.handlers.warn_and_continue),
        wds.to_tuple(("image",)),
    )

    return ds
