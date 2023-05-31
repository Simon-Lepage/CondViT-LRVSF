import click
import os
import argparse
from pathlib import Path

from glob import glob
from tqdm import tqdm
import pickle

import pandas as pd
import numpy as np
import faiss


_NDISTRS = [
    0,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
    500000,
    1000000,
    2000015,
]
_KMAX = 100


def init_bootstrap_infos(args):
    f = open(os.path.join(args["dataset_root"], "bootstrap_IDs.pkl"), "rb")
    BDICT = pickle.load(f)
    f.close()

    N = len(BDICT["test_subsets"])

    DIST_KEYS = ["dist_0_subsets"] + [k for k in BDICT if k.startswith("dist_")]
    BDICT["dist_0_subsets"] = [np.array([]) for _ in range(N)]

    for k in DIST_KEYS:
        assert len(BDICT[k]) == N

    return DIST_KEYS, BDICT, N


def build_index(*args):
    index = faiss.IndexFlatIP(args[0].shape[1])
    for a in args:
        index.add(a)
    index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    return index


def pad(array, L, value):
    # Pad last axis
    s = list(array.shape)

    if s[-1] < L:
        s[-1] = L - s[-1]
        array = np.concatenate([array, np.ones(s) * value], axis=len(array.shape) - 1)

    return array


def get_metrics(q_embs, g_embs, dist_embs, q_metas, g_metas, dist_metas, weights=None):
    accumulator = {}

    is_weighted = weights is not None

    # Grab gallery categorical infos
    gallery_categories = pd.concat(
        [g_metas, dist_metas], axis=0, ignore_index=True
    ).CATEGORY
    targets_categories = g_metas.CATEGORY.values

    # Index with all gallery images and selected dists.
    index = build_index(g_embs, dist_embs)
    results = index.search(q_embs, _KMAX)

    hits = results[1] == np.arange(q_embs.shape[0])[:, None]
    assert (hits.sum(axis=1) <= 1).all()
    hits_cumsum = hits.cumsum(axis=1)
    cat_hits = (
        np.take(gallery_categories.values, results[1]) == targets_categories[:, None]
    )

    # Repeat
    if is_weighted:
        hits = hits.repeat(weights, axis=0)
        hits_cumsum = hits_cumsum.repeat(weights, axis=0)
        cat_hits = cat_hits.repeat(weights, axis=0)

    # Update store
    accumulator["unfilt_r@k"] = hits_cumsum.mean(axis=0)
    accumulator["unfilt_p@k"] = (hits_cumsum / (np.arange(_KMAX) + 1)[None, :]).mean(
        axis=0
    )
    accumulator["unfilt_rank"] = pad(
        np.where(hits)[1], hits.shape[0], np.inf
    )  # Will be shorter than NQueries -> Keep
    accumulator["unfilt_p@r=1"] = 1 / (accumulator["unfilt_rank"] + 1)

    cat_acc = cat_hits.cumsum(axis=1) / np.arange(1, _KMAX + 1)[None, :]
    accumulator["unfilt_acc@k"] = cat_acc.mean(axis=0)

    r_acc = []
    hits_acc = []
    filt_weights = []
    for category in g_metas.CATEGORY.unique():
        prod_mask = (g_metas.CATEGORY == category).values
        if prod_mask.sum() == 0:
            continue  # No objects of this category in this split, continue.

        sub_q = q_embs[prod_mask]
        sub_g = g_embs[prod_mask]
        sub_d = dist_embs[dist_metas.CATEGORY == category]

        index = build_index(sub_g, sub_d)
        results = index.search(sub_q, min(_KMAX, index.ntotal))

        hits = results[1] == np.arange(sub_q.shape[0])[:, None]
        hits = pad(hits, _KMAX, 0)
        hits_acc.append(hits)

        assert (hits.sum(axis=1) <= 1).all()

        _r = np.cumsum(hits, axis=1)
        _r = pad(_r, _KMAX, 1)  # Might be useless ?
        r_acc.append(_r)

        if is_weighted:
            filt_weights.append(weights[prod_mask])

    r_acc = np.concatenate(r_acc, axis=0)
    hits_acc = np.concatenate(hits_acc)

    if is_weighted:
        filt_weights = np.concatenate(filt_weights)
        r_acc = r_acc.repeat(filt_weights, axis=0)
        hits_acc = hits_acc.repeat(filt_weights, axis=0)

    # Update store
    accumulator["filt_r@k"] = r_acc.mean(axis=0)
    accumulator["filt_p@k"] = (r_acc / (np.arange(_KMAX) + 1)[None, :]).mean(axis=0)
    accumulator["filt_rank"] = pad(np.where(hits_acc)[1], hits_acc.shape[0], np.inf)
    accumulator["filt_p@r=1"] = 1 / (accumulator["filt_rank"] + 1)

    return accumulator


def evaluate(args):
    # ======= EMBEDDINGS AND METADATA LOADING =======

    assert os.path.isdir(
        args["embeddings_folder"]
    ), f"Folder {args['embeddings_folder']} not found."

    print("Reading embeddings.")
    dist_embs = pd.concat(
        [
            pd.read_parquet(p)
            for p in tqdm(
                glob(os.path.join(args["embeddings_folder"], "dist*.parquet"))
            )
        ]
    ).set_index("url", drop=True)
    val_embs = pd.read_parquet(
        os.path.join(args["embeddings_folder"], "products_embs.parquet")
    ).set_index("url", drop=True)

    print("Reading metadata.")
    dist_metas = pd.read_parquet(
        os.path.join(args["dataset_root"], "distractors_test_metadata.parquet")
    )
    dist_metas = dist_metas[dist_metas.SPLIT == "test_gallery"]

    products_metas = pd.concat(
        [
            pd.read_parquet(p)
            for p in glob(
                os.path.join(args["dataset_root"], "products*metadata.parquet")
            )
        ]
    )
    products_metas = products_metas[products_metas.SPLIT.str.contains("test")]

    q_metas = products_metas[products_metas.SPLIT == "test_query"].sort_values(
        "PRODUCT_ID"
    )
    g_metas = products_metas[products_metas.SPLIT == "test_gallery"].sort_values(
        "PRODUCT_ID"
    )
    # Check that i-th query -> i-th gallery
    assert (q_metas.PRODUCT_ID.values == g_metas.PRODUCT_ID.values).all()

    # Full np arrays
    # Ensure same order.
    print("Stacking embeddings.")
    dist_embs = np.stack(dist_embs.loc[dist_metas.URL].embeddings.values)
    q_embs = np.stack(val_embs.loc[q_metas.URL].embeddings.values)
    g_embs = np.stack(val_embs.loc[g_metas.URL].embeddings.values)

    for a in [dist_embs, q_embs, g_embs]:
        faiss.normalize_L2(a)

    metrics_store = {}

    print("Computing full testset metrics.")
    metrics_store = {}
    for ndist in tqdm(_NDISTRS):
        subdist = dist_embs[:ndist]
        subdist_metas = dist_metas.iloc[:ndist]

        metrics_store[f"fullset+{subdist.shape[0]}"] = get_metrics(
            q_embs, g_embs, subdist, q_metas, g_metas, subdist_metas
        )

    print("Computing bootstrapped metrics.")
    DIST_KEYS, BDICT, N = init_bootstrap_infos(args)

    metrics_store["bootstrap"] = {}

    for n in tqdm(range(N)):
        boot_test_ids = BDICT["test_subsets"][n]

        # Subset Test : use .isin because we only want one of each.
        boot_metas_loc = q_metas.PRODUCT_ID.isin(boot_test_ids).values  # Boolean mask
        boot_q_metas = q_metas.iloc[boot_metas_loc]
        boot_g_metas = g_metas.iloc[boot_metas_loc]
        assert (boot_q_metas.PRODUCT_ID.values == boot_g_metas.PRODUCT_ID.values).all()
        boot_q_embs = q_embs[
            boot_metas_loc
        ]  # Can use this because embs were created to have the same order (.loc[] L.133)
        boot_g_embs = g_embs[boot_metas_loc]

        # Repeat Weights
        unique, counts = np.unique(boot_test_ids, return_counts=True)
        v_2_c = dict(zip(unique, counts))
        weights = np.array([v_2_c[p] for p in boot_q_metas.PRODUCT_ID.values])
        assert sum(weights) == boot_test_ids.shape[0]

        for dk in DIST_KEYS:
            boot_dist_ids = BDICT[dk][n]

            # Subset distractors : get ilocs for metas and embs
            boot_dists_iloc = dist_metas.index.get_indexer(boot_dist_ids)
            boot_subdists_metas = dist_metas.iloc[
                boot_dists_iloc
            ]  # Can contain multiple versions of an image
            boot_subdists_embs = dist_embs[boot_dists_iloc]

            out = get_metrics(
                boot_q_embs,
                boot_g_embs,
                boot_subdists_embs,
                boot_q_metas,
                boot_g_metas,
                boot_subdists_metas,
                weights,
            )

            for k in out:
                if not dk in metrics_store["bootstrap"]:
                    metrics_store["bootstrap"][dk] = {}
                if not k in metrics_store["bootstrap"][dk]:
                    metrics_store["bootstrap"][dk][k] = np.zeros(
                        shape=(N, out[k].shape[0])
                    )
                metrics_store["bootstrap"][dk][k][n, :] = out[k]  # Insert as line

    return metrics_store


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_folder", type=Path, required=True)
    parser.add_argument(
        "--dataset_root", type=str, default=os.path.expanduser("~/DATA/LRVSF")
    )
    parser.add_argument("--force_regen", action="store_true")
    args = vars(parser.parse_args())

    return args


def check_dst(dst, force_regen):
    if os.path.exists(dst):
        if force_regen or click.confirm(
            f"File {dst} already exists. Do you want to continue?"
        ):
            os.remove(dst)
        else:
            exit(-1)


if __name__ == "__main__":
    args = get_args()

    dst = os.path.join(args["embeddings_folder"].parent, "metrics.pkl")
    check_dst(dst, args["force_regen"])

    # Evaluate
    store = evaluate(args)

    # Write to file
    with open(dst, "wb") as f:
        pickle.dump(store, f)
