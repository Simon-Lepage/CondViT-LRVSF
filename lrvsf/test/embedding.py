import argparse
import os
import click
from pathlib import Path
from glob import glob
import torch
import shutil
import pandas as pd

import lrvsf


def postpro(urls, embeddings):
    urls = sum(urls, [])
    embeddings = [list(f.chunk(f.shape[0], dim=0)) for f in embeddings]
    embeddings = sum(embeddings, [])
    embeddings = [f.numpy().flatten() for f in embeddings]
    return urls, embeddings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--force_regen", action="store_true")
    parser.add_argument(
        "--dataset_root", type=str, default=os.path.expanduser("~/DATA/LRVSF")
    )
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    args = vars(args)

    args["exp_name"] = exp_name_from_save_path(args["save_path"])
    args["embs_exp_folder"] = os.path.join(args["save_path"].parent, "embs")

    return args


def exp_name_from_save_path(s):
    return s.parts[-2]


def check_experiment_folder(f, force_regen):
    if os.path.exists(f):
        if force_regen or click.confirm(
            f"Folder {f} already exists. Do you want to continue?"
        ):
            shutil.rmtree(f)
        else:
            exit(-1)

    os.mkdir(f)


def embed(rank, worldsize, args):
    torch.set_grad_enabled(False)

    model, use_conditioning = lrvsf.chkpt_utils.load_chkpt(args["save_path"])
    model.eval()
    model.to(rank)

    tfs = lrvsf.data.transforms.valid_tf((224, 224))

    shards_list = glob(os.path.join(args["dataset_root"], "TEST/dist*.parquet"))
    for shard in shards_list:
        filename = shard.split("/")[-1].split(".")[0]

        # Destination files
        pq_file = f"{args['embs_exp_folder']}/{filename}.parquet"
        feather_file = f"{args['embs_exp_folder']}/{filename}.feather"

        if os.path.exists(feather_file):
            continue  # Another process is already working on it

        with open(feather_file, "a") as f:
            f.write("")

        print("Processing", shard)
        df = pd.read_parquet(shard)
        ds = lrvsf.data.parquetdatasets.URLandImgs(df, tfs)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=args["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count() // 2 - 1,
        )

        embeddings = []
        urls = []

        for u, imgs in dl:
            imgs = imgs.to(rank)
            emb = torch.nn.functional.normalize(model(imgs))

            embeddings.append(emb.cpu())
            urls.append(list(u))

        urls, embeddings = postpro(urls, embeddings)

        pd.DataFrame({"url": urls, "embeddings": embeddings}).to_parquet(pq_file)

    if rank != 0:
        exit(0)

    print("Processing validation products.")

    pq_file = f"{args['embs_exp_folder']}/products_embs.parquet"
    df_imgs = pd.read_parquet(os.path.join(args["dataset_root"], "TEST/prods.parquet"))
    df_metas = pd.concat(
        [
            pd.read_parquet(
                os.path.join(args["dataset_root"], "products_simple_metadata.parquet")
            ),
            pd.read_parquet(
                os.path.join(args["dataset_root"], "products_complex_metadata.parquet")
            ),
        ]
    )
    df_metas = df_metas[df_metas.SPLIT.isin(["test_query", "test_gallery"])]

    ds = lrvsf.data.parquetdatasets.TestProducts(
        df_imgs, df_metas, tfs, use_conditioning
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=os.cpu_count() // 2 - 1,
        collate_fn=lrvsf.data.parquetdatasets.collate_products_none
    )

    embeddings = []
    urls = []

    for p_batch in dl:
        # Unpack batch : q_cat might be none
        g_url, q_url, g_img, q_img, q_cat = p_batch

        for u, batch in [
            (g_url, (g_img,)),
            (q_url, (q_img, q_cat)),
        ]:
            batch = [b.to(rank) for b in batch if b is not None]
            emb = torch.nn.functional.normalize(model(*batch))

            embeddings.append(emb.cpu())
            urls.append(u)

    urls, embeddings = postpro(urls, embeddings)
    pd.DataFrame({"url": urls, "embeddings": embeddings}).to_parquet(pq_file)


if __name__ == "__main__":
    args = get_args()
    check_experiment_folder(args["embs_exp_folder"], args["force_regen"])

    # Launch distributed embedding
    lrvsf.dist_utils.spawn_fn(embed, torch.cuda.device_count(), args)

    # Remove .feather files
    for file in glob(os.path.join(args["embs_exp_folder"], "*.feather")):
        os.remove(file)
