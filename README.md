# Weakly-Supervised Conditional Embedding for Referred Visual Search

**[CRITEO AI Lab](https://ailab.criteo.com)** $\times$ **[ENPC](https://imagine-lab.enpc.fr)**

[Simon Lepage](https://simon-lepage.github.io), Jérémie Mary, [David Picard](https://davidpicard.github.io)

[[`Paper`](https://arxiv.org/abs/2306.02928)] 
[[`Demo`](https://huggingface.co/spaces/Slep/CondViT-LRVSF-Demo)] 
[[`Dataset`](https://huggingface.co/datasets/Slep/LAION-RVS-Fashion)] 
[[`BibTeX`](#citing-our-work)]

---

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

### **Run the training**

Use the following commands to train a model and evaluate it. Additional options can be found in the scripts. Training a ViT-B/32 on 2 Nvidia V100 GPUs should take ~6h.

```shell
python main.py --architecture B32 --batch_size 180 --conditioning --run_name CondViTB32
python lrvsf/test/embedding.py --save_path ./saves/CondViTB32_*/best_validation_model.pth
python lrvsf/test/metrics.py --embeddings_folder ./saves/CondViTB32_*/embs/
```

---

## Citing our work

To cite our work, please use the following BibTeX entry : 
```
@article{lepage2023condvit,
  title={Weakly-Supervised Conditional Embedding for Referred Visual Search},
  author={Lepage, Simon and Mary, Jérémie and Picard, David},
  journal={arXiv:2306.02928},
  year={2023}
}
```
