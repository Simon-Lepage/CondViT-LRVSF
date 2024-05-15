<div align="center">

# Conditional ViT Training - LRVS-F
Introduced in ***LRVSF-Fashion: Extending Visual Search with Referring Instructions***

<a href="https://simon-lepage.github.io"><strong>Simon Lepage</strong></a>
—
<strong>Jérémie Mary</strong>
—
<a href=https://davidpicard.github.io><strong>David Picard</strong></a>

<a href=https://ailab.criteo.com>CRITEO AI Lab</a>
&
<a href=https://imagine-lab.enpc.fr>ENPC</a>
</div>

<p align="center">
    <a href="https://arxiv.org/abs/2306.02928">
        <img alt="ArXiV Badge" src="https://img.shields.io/badge/arXiv-2306.02928-b31b1b.svg">
    </a>
</p>

<div align="center">
<div id=links>

|Data|Code|Models|Spaces|
|:-:|:-:|:-:|:-:|
|[Full Dataset](https://huggingface.co/datasets/Slep/LAION-RVS-Fashion)|[Training Code](https://github.com/Simon-Lepage/CondViT-LRVSF)|[Categorical Model](https://huggingface.co/Slep/CondViT-B16-cat)|[LRVS-F Leaderboard](https://huggingface.co/spaces/Slep/LRVSF-Leaderboard)|
|[Test set](https://zenodo.org/doi/10.5281/zenodo.11189942)|[Benchmark Code](https://github.com/Simon-Lepage/LRVSF-Benchmark)|[Textual Model](https://huggingface.co/Slep/CondViT-B16-txt)|[Demo](https://huggingface.co/spaces/Slep/CondViT-LRVSF-Demo)|
</div>
</div>

To cite our work, please use the following BibTeX entry : 
```bibtex
@article{lepage2023lrvsf,
  title={LRVS-Fashion: Extending Visual Search with Referring Instructions},
  author={Lepage, Simon and Mary, Jérémie and Picard, David},
  journal={arXiv:2306.02928},
  year={2023}
}
```

---

## Overview
![Method](./assets/method.png?raw=true)

**CondViT** computes conditional image embeddings, to extract specific features out of complex images. This codebase shows how to train it on [LAION — Referred Visual Search — Fashion](https://huggingface.co/datasets/Slep/LAION-RVS-Fashion), with clothing categories. It can easily be modified to use free text embeddings as conditioning, such as BLIP2 captions provided in the dataset.

**Categorical CondViT Results :**
![Categorical Results](./assets/results.png?raw=true)
**Textual CondViT Results :**
![Textual Results](./assets/textual_results.png?raw=true)


## **Usage**

### **Install the project and its requirements**
```shell
git clone git@github.com:Simon-Lepage/CondViT-LRVSF.git
cd CondViT-LRVSF; pip install -e .
```

### **Prepare [CLIP](https://github.com/openai/CLIP) saves in `models` folder.**
```python
import clip 

torch.save(clip.load("ViT-B/32")[0].visual.state_dict(), "models/CLIP_B32_visual.pth")
torch.save(clip.load("ViT-B/16")[0].visual.state_dict(), "models/CLIP_B16_visual.pth")
```

### **Download [LRVS-F](https://huggingface.co/datasets/Slep/LAION-RVS-Fashion)**

 We recommend [img2dataset](https://github.com/rom1504/img2dataset) to download the images. 
- **Training products** should be stored in tarfiles in a `TRAIN` folder, as the training uses webdataset. For each product, its image should be stored as `{PRODUCT_ID}.{i}.jpg` and accompagnied by a `<PRODUCT_ID>.json` file with each `i` as keys of metadata. This will require reorganising the tarfiles natively produced by img2dataset.
    <details><summary><b>Exemple</b></summary>
        
    ```
    ...
    230537.0.jpg
    230537.1.jpg
    230537.json => {
        "0": {
            "URL": "https://img01.ztat.net/article/LE/22/2G/09/CQ/11/LE222G09C-Q11@6.jpg?imwidth=762",
            "TYPE": "COMPLEX",
            "SPLIT": "train",
            [...]
        },
        "1": {
            "URL": "https://img01.ztat.net/article/LE/22/2G/09/CQ/11/LE222G09C-Q11@10.jpg?imwidth=300&filter=packshot",
            "TYPE": "SIMPLE",
            "SPLIT": "train",
            "CATEGORY": "Lower Body",
            "blip2_caption1": "levi's black jean trousers - skinny fit",
            [ ... ]
        }
    }
    ...
    ```
    </details>

- **Validation** products should also be stored in `VALID/prods.tar` following the same format. Validation distractors should be stored in `VALID/dist_{i}.tar` as `{ID}.jpg`, `{ID}.json`. The JSON file should directly contain metadata.
    <details><summary><b>Exemple</b></summary>
        
    ```
    ...
    989760.jpg
    989760.json => {
        "URL": "https://img01.ztat.net/article/spp-media-p1/0dd705f32f9e4895810d291c76de5ea2/1661e4ee07f342dcb168fed3ab81e78e.jpg?imwidth=300&filter=packshot",
        "CATEGORY": "Lower Body",
        "SPLIT": "val_gallery"
        [...]
    }
    ...
    ```
    </details>
- **Test** data should be stored as `TEST/dist_{i}.parquet` and `TEST/prods.parquet` files. Their index should be `url`, and have a single `jpg` column containing the images as bytes.

### **Train the model**

Use the following commands to train a model and evaluate it. Additional options can be found in the scripts. Training a ViT-B/32 on 2 Nvidia V100 GPUs should take ~6h.

```shell
python main.py --architecture B32 --batch_size 180 --conditioning --run_name CondViTB32
python lrvsf/test/embedding.py --save_path ./saves/CondViTB32_*/best_validation_model.pth
python lrvsf/test/metrics.py --embeddings_folder ./saves/CondViTB32_*/embs/
```
