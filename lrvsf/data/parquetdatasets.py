from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
import torch 

from lrvsf.constants import categories


class URLandImgs(Dataset):
    def __init__(self, data, tf):
        self.data = data
        self.tf = tf

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row.name, self.tf(Image.open(BytesIO(row.jpg)).convert("RGB"))


class TestProducts(Dataset):
    def __init__(self, imgs, meta, tf, use_conditioning):
        self.imgs = imgs
        self.meta = meta
        self.tf = tf
        self.use_conditioning = use_conditioning

        self.prod_ids = meta.PRODUCT_ID.unique().tolist()
        self.groups = self.meta.groupby("PRODUCT_ID")

    def __len__(self):
        return len(self.prod_ids)

    def __getitem__(self, index):
        group = self.groups.get_group(self.prod_ids[index])
        assert group.shape[0] == 2
        images = self.imgs.loc[group.URL]

        gallery_meta = group[group.SPLIT.str.contains("gallery")].iloc[0]
        gallery_img = Image.open(BytesIO(images.loc[gallery_meta.URL].jpg)).convert(
            "RGB"
        )

        query_meta = group[group.SPLIT.str.contains("query")].iloc[0]
        query_img = Image.open(BytesIO(images.loc[query_meta.URL].jpg)).convert("RGB")

        return [
            gallery_meta.URL,
            query_meta.URL,
            self.tf(gallery_img),
            self.tf(query_img),
        ] + [categories.index(gallery_meta.CATEGORY) if self.use_conditioning else None]


def collate_products_none(batch):
    gurl, qurl, gimg, qimg, qcat = zip(*batch)

    return [
        list(gurl), 
        list(qurl), 
        torch.stack(gimg),
        torch.stack(qimg), 
        None if qcat[0] is None else torch.tensor(qcat)
    ]