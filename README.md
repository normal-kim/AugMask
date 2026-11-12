# AugMask

Official code for **AugMask: Training Diffusion Models on Incomplete Tabular Data via Stochastic Augmentation and Masking** (ICML 2026).
Jungkyu Kim, Taeyoung Park, Kibok Lee. [[arXiv]](https://arxiv.org/abs/2606.03347) [[OpenReview]](https://openreview.net/forum?id=Ot8lh7vvGe)

## 💡 Why AugMask

**Diffusion models for tabular data are built for complete rows**, but real tables have missing entries. The usual workarounds fail:

- **Constant fills become data.** A model trained on mean-filled rows *treats the fills as observed values* and learns to reproduce them. Below, the filled rows collapse onto a cross at the column means.
- **Masking the loss is not enough.** Filled entries left out of the loss are *still read as inputs*, so the model learns from rows that real data rarely contains, such as a hot day paired with the average dew point.

AugMask fixes both:

1. **Augment.** A missing entry is *unrealized data that could take many different values*, so AugMask *samples* it instead of imputing it deterministically. Each missing entry $`x_j`$ is drawn from a conditional model of its column, $`x_j \sim \hat{p}_j(x_j \mid x_{\mathrm{obs}}, m)`$: the augmenter receives the row's observed values $`x_{\mathrm{obs}}`$ and its missingness mask $`m`$ as inputs. This repository uses LGBM-S, lightweight per-column LightGBM models, as the augmenter; the sampled rows resemble the complete data (right panel).
2. **Mask.** The denoising loss is computed on observed entries only, so the sampled values serve as *context, never as targets*.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/overview_dark.png">
  <img src="assets/overview_light.png" alt="Mean filling collapses rows onto a cross at the column means; augmented fills follow the joint distribution. Inputs are filled, and the loss uses observed entries only." width="100%">
</picture>

AugMask needs no change to the model itself. This repository applies it to three diffusion models: [TabDDPM](https://proceedings.mlr.press/v202/kotelnikov23a.html) (ICML 2023; [code](https://github.com/yandex-research/tab-ddpm)), [CDTD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/61f4e5747b1b753cb35546b15d981f76-Abstract-Conference.html) (ICLR 2025; [code](https://github.com/muellermarkus/cdtd)) and [TabDiff](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html) (ICLR 2025; [code](https://github.com/MinkaiXu/TabDiff)).

## 📦 Install

Python 3.10–3.12.

```bash
conda env create -f environment.yml && conda activate augmask   # the exact environment used for the results
# or
pip install -e ".[all]"                                          # everything (pip install -e . for the augmentation alone)
```

## 🧩 Use AugMask with your own model

```python
from augmask import Encoded, LGBMSAugmenter

train = Encoded(X_num, X_cat, M_num, M_cat)
fills = [LGBMSAugmenter(draw=k).augment(train, [], n_categories)[0] for k in range(5)]

# train your denoiser on fills[step % 5], with the loss restricted to observed entries
loss = (per_entry_loss * mask).sum() / mask.sum()
```

The augmenter takes an encoded table with `n` rows:

| Input | Shape | Type | Content |
|---|---|---|---|
| `X_num` | `(n, d_num)` | float | numerical columns, quantile-normal transformed; `NaN` where missing |
| `X_cat` | `(n, d_cat)` | int | categorical columns, integer encoded as `0 … K_j−1`; `−1` where missing |
| `M_num`, `M_cat` | same as above | bool | `True` where observed |
| `n_categories` | `d_cat` | list of int | number of categories `K_j` of each categorical column |

`augment` returns `X_num` and `X_cat` with every missing entry filled; each `draw` gives an independent fill, with the observed entries unchanged. In this repository, the backbones receive one fill per training step: `X_num` (float32, standardized) and `X_cat` (int64 codes), together with the masks. The target is one of the columns (the first numerical column for regression, the first categorical column for classification).

### Your own augmenter

Any object with an `augment` method can replace LGBM-S:

```python
from augmask.run import run

class MyAugmenter:
    def augment(self, train, others, n_categories):
        # fit on `train` (an Encoded table); return [(X_num, X_cat), ...] for `train` and each of `others`,
        # with every missing entry filled: floats in X_num, codes 0 … K_j−1 in X_cat.
        # It is called once per fill (5 by default), so return a new random draw on each call.
        ...

run("adult", "mcar", 0.5, "tabdiff-augmask", augmenter=MyAugmenter())
```

## 🚀 Run

Datasets are downloaded from UCI on first use.

```bash
# one method on one scenario: train, sample and evaluate  ->  runs/<dataset>/<mechanism>/p<p>/seed<s>/<method>.{csv,json}
python -m augmask.run --dataset adult --mechanism mcar --p 0.5 --method cdtd-augmask

# all experiments (6 datasets x 5 missing rates x 3 seeds x 14 methods) followed by the summary tables
bash scripts/run_all.sh 0                  # on GPU 0
bash scripts/run_all.sh 1 beijing news     # or split the datasets across GPUs and servers,
python scripts/make_tables.py              # then summarise the merged runs/
```

Methods are `{tabddpm,cdtd,tabdiff}-{noaug,meanmask,augfull,augmask}`, plus the baselines `forestdiff` and `miwae`.

| Strategy | Missing inputs filled with | Loss on |
|---|---|---|
| `noaug` | train mean / mode | all entries |
| `meanmask` | train mean / mode | observed entries (similar to [MissDiff](https://arxiv.org/abs/2307.00467); Ouyang et al., 2023) |
| `augfull` (ours) | samples from the augmenter (LGBM-S) | all entries |
| `augmask` (ours) | samples from the augmenter (LGBM-S) | observed entries |

**Alternating fills.** For improved performance, this implementation gives `augfull` and `augmask` 5 independent fills of the missing inputs and alternates between them during training, one per step (`--fills 5`, the default). `--fills 1` uses the single fill of the paper. The gain is largest at high missing rates.

**Masks and fills.** The first run of a scenario saves its missingness masks and augmented fills to `artifacts/<dataset>/<mechanism>/p<p>/seed<s>.npz`; later runs of the same scenario reuse them, so LGBM-S runs once per scenario rather than once per method.

**Runtime.** LGBM-S takes a few seconds per fill (fit on the train split, fill train and validation; AMD Ryzen 9 5950X, 32 threads); the default 5 fills take about five times as long, once per scenario:

| Missing | Adult | Default | Shoppers | Magic | Beijing | News |
|---|---:|---:|---:|---:|---:|---:|
| 10% | 3.5 s | 2.0 s | 1.3 s | 0.3 s | 2.9 s | 5.7 s |
| 50% | 4.5 s | 2.3 s | 1.5 s | 0.3 s | 4.1 s | 4.9 s |
| 90% | 6.0 s | 2.0 s | 1.1 s | 0.3 s | 5.8 s | 2.9 s |

## 📊 Reference results

Adult and Beijing, MCAR with 50% and 90% missing, one seed, averaged over the four scenarios (×100, best in bold). Utility is downstream task performance: a model trained on synthetic rows and tested on real ones, relative to training on real rows.

- **Missing-aware:** [ForestDiff](https://proceedings.mlr.press/v238/jolicoeur-martineau24a.html) (AISTATS 2024) and [MIWAE](https://proceedings.mlr.press/v97/mattei19a.html) (ICML 2019).
- **Missing-unaware**, trained with the four strategies above: [TabDDPM](https://proceedings.mlr.press/v202/kotelnikov23a.html) (ICML 2023), [CDTD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/61f4e5747b1b753cb35546b15d981f76-Abstract-Conference.html) (ICLR 2025) and [TabDiff](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html) (ICLR 2025).

| Method | Shape ↑ | Trend ↑ | α-Precision ↑ | β-Recall ↑ | Utility ↑ |
|---|---:|---:|---:|---:|---:|
| TabDDPM + NoAug | 68.2 | 54.7 | 45.1 | 8.7 | 66.0 |
| TabDDPM + MeanMask | 94.9 | 88.5 | 91.9 | 28.3 | 74.5 |
| TabDDPM + AugFull (ours) | 95.7 | 90.2 | 93.5 | 31.7 | 75.6 |
| TabDDPM + AugMask (ours) | 94.1 | 88.5 | 90.8 | 35.6 | 77.9 |
| CDTD + NoAug | 68.5 | 55.7 | 44.5 | 9.1 | 66.2 |
| CDTD + MeanMask | 93.6 | 84.5 | 93.2 | 23.0 | 73.6 |
| CDTD + AugFull (ours) | 95.5 | 91.0 | 91.2 | 33.2 | 77.2 |
| CDTD + AugMask (ours) | 97.8 | 93.4 | 98.0 | 37.8 | 76.7 |
| TabDiff + NoAug | 66.9 | 54.3 | 43.8 | 9.0 | 67.9 |
| TabDiff + MeanMask | 96.6 | 88.8 | 95.0 | 29.3 | 73.3 |
| TabDiff + AugFull (ours) | 95.7 | 91.4 | 93.1 | 32.3 | 76.6 |
| TabDiff + AugMask (ours) | **98.1** | **94.4** | **98.4** | **39.1** | **79.1** |
| ForestDiff | 69.7 | 66.7 | 65.6 | 22.4 | 66.4 |
| MIWAE | 97.5 | 85.2 | 97.1 | 29.1 | 72.4 |

<details>
<summary>Per scenario</summary>

**Adult, 50% missing**

| Method | Shape ↑ | Trend ↑ | α-Precision ↑ | β-Recall ↑ | Utility ↑ |
|---|---:|---:|---:|---:|---:|
| TabDDPM + NoAug | 81.9 | 71.9 | 51.5 | 15.5 | 95.9 |
| TabDDPM + MeanMask | 94.7 | 89.7 | 90.8 | 30.9 | 97.3 |
| TabDDPM + AugFull (ours) | 97.2 | 92.3 | 97.3 | 35.1 | 96.8 |
| TabDDPM + AugMask (ours) | 96.2 | 92.6 | 95.7 | 43.3 | 97.2 |
| CDTD + NoAug | 83.3 | 74.2 | 54.1 | 16.6 | 96.4 |
| CDTD + MeanMask | 95.3 | 92.0 | 93.1 | 34.4 | 96.8 |
| CDTD + AugFull (ours) | 96.7 | 94.0 | 94.8 | 37.9 | **97.9** |
| CDTD + AugMask (ours) | 98.9 | 97.2 | 97.3 | **46.0** | 97.9 |
| TabDiff + NoAug | 81.0 | 72.3 | 50.8 | 15.9 | 95.1 |
| TabDiff + MeanMask | 96.2 | 93.6 | 96.1 | 35.7 | 97.5 |
| TabDiff + AugFull (ours) | 97.1 | 94.2 | 98.1 | 36.5 | 97.2 |
| TabDiff + AugMask (ours) | **99.1** | **97.8** | **99.4** | 45.4 | 97.7 |
| ForestDiff | 90.7 | 91.3 | 95.4 | 34.8 | 97.0 |
| MIWAE | 97.3 | 94.9 | 98.3 | 30.8 | 95.5 |

**Adult, 90% missing**

| Method | Shape ↑ | Trend ↑ | α-Precision ↑ | β-Recall ↑ | Utility ↑ |
|---|---:|---:|---:|---:|---:|
| TabDDPM + NoAug | 66.3 | 46.2 | 10.9 | 1.8 | 71.7 |
| TabDDPM + MeanMask | 94.6 | 85.4 | 92.9 | 11.8 | 93.7 |
| TabDDPM + AugFull (ours) | 94.6 | 85.0 | 95.8 | 13.6 | 92.4 |
| TabDDPM + AugMask (ours) | 93.3 | 85.0 | 90.1 | 30.0 | 94.9 |
| CDTD + NoAug | 69.0 | 50.1 | 11.9 | 1.9 | 70.0 |
| CDTD + MeanMask | 93.2 | 75.3 | 90.3 | 9.1 | 87.5 |
| CDTD + AugFull (ours) | 93.9 | 84.2 | 91.2 | 14.0 | 92.6 |
| CDTD + AugMask (ours) | **97.9** | 90.9 | **99.1** | 30.9 | 94.0 |
| TabDiff + NoAug | 65.9 | 47.1 | 10.5 | 1.8 | 78.9 |
| TabDiff + MeanMask | 95.5 | 87.2 | 90.9 | 10.3 | 91.6 |
| TabDiff + AugFull (ours) | 94.4 | 85.1 | 94.9 | 14.1 | 93.8 |
| TabDiff + AugMask (ours) | 97.4 | **91.1** | 97.5 | **33.0** | **95.4** |
| ForestDiff | 36.3 | 26.9 | 1.4 | 0.3 | 72.3 |
| MIWAE | 97.6 | 89.1 | 96.5 | 14.2 | 92.0 |

**Beijing, 50% missing**

| Method | Shape ↑ | Trend ↑ | α-Precision ↑ | β-Recall ↑ | Utility ↑ |
|---|---:|---:|---:|---:|---:|
| TabDDPM + NoAug | 74.2 | 64.1 | 70.3 | 15.9 | 50.5 |
| TabDDPM + MeanMask | 95.3 | 92.1 | 92.0 | 37.4 | 58.0 |
| TabDDPM + AugFull (ours) | 97.6 | 94.8 | 97.2 | 39.8 | 62.3 |
| TabDDPM + AugMask (ours) | 97.4 | 93.0 | 94.9 | 39.0 | 65.5 |
| CDTD + NoAug | 72.2 | 64.8 | 68.1 | 16.6 | 56.4 |
| CDTD + MeanMask | 95.2 | 92.7 | 97.0 | 39.6 | 64.6 |
| CDTD + AugFull (ours) | 97.5 | 95.5 | 95.2 | 42.1 | 66.0 |
| CDTD + AugMask (ours) | 98.5 | 96.0 | 98.5 | **42.9** | 62.8 |
| TabDiff + NoAug | 72.2 | 64.4 | 69.2 | 16.4 | 53.5 |
| TabDiff + MeanMask | 97.2 | 94.7 | 97.9 | 38.6 | 59.9 |
| TabDiff + AugFull (ours) | 97.7 | 95.8 | 95.8 | 40.5 | 64.1 |
| TabDiff + AugMask (ours) | **98.8** | **96.3** | **98.9** | 40.8 | **67.9** |
| ForestDiff | 91.9 | 90.6 | 93.1 | 35.1 | 62.4 |
| MIWAE | 97.8 | 87.1 | 97.2 | 39.2 | 58.3 |

**Beijing, 90% missing**

| Method | Shape ↑ | Trend ↑ | α-Precision ↑ | β-Recall ↑ | Utility ↑ |
|---|---:|---:|---:|---:|---:|
| TabDDPM + NoAug | 50.4 | 36.6 | 47.7 | 1.6 | 46.1 |
| TabDDPM + MeanMask | 94.9 | 86.6 | 91.7 | 33.3 | 49.1 |
| TabDDPM + AugFull (ours) | 93.6 | 88.6 | 83.8 | 38.1 | 50.9 |
| TabDDPM + AugMask (ours) | 89.6 | 83.4 | 82.7 | 30.1 | 53.9 |
| CDTD + NoAug | 49.5 | 33.7 | 43.8 | 1.0 | 42.0 |
| CDTD + MeanMask | 90.5 | 78.1 | 92.3 | 9.0 | 45.4 |
| CDTD + AugFull (ours) | 93.7 | 90.4 | 83.6 | **38.8** | 52.2 |
| CDTD + AugMask (ours) | 95.6 | 89.6 | 97.0 | 31.6 | 52.1 |
| TabDiff + NoAug | 48.6 | 33.7 | 44.9 | 1.8 | 44.3 |
| TabDiff + MeanMask | 97.4 | 79.8 | 95.2 | 32.3 | 44.3 |
| TabDiff + AugFull (ours) | 93.8 | 90.8 | 83.7 | 38.2 | 51.2 |
| TabDiff + AugMask (ours) | 97.0 | **92.6** | **98.0** | 37.3 | **55.6** |
| ForestDiff | 59.7 | 58.1 | 72.4 | 19.6 | 33.7 |
| MIWAE | **97.5** | 69.5 | 96.2 | 32.3 | 43.9 |

</details>

## 🔧 Pipeline

1. **Collect** the six UCI tables (Adult, Default, Shoppers, Magic, Beijing, News).
2. **Split** each table 60/20/20 into train, validation and test.
3. **Mask** train and validation entries (the target included) completely at random (MCAR) or depending on the other columns (MAR); the test split stays complete.
4. **Encode**, using the observed train entries: numerical columns are quantile-normal transformed, categorical columns are integer encoded.
5. **Fill** the missing entries with the train mean / mode, or *sample* them with the augmenter (LGBM-S by default; 5 fills, alternated during training).
6. **Standardize** the numerical columns.
7. **Train** the backbone and sample as many rows as the train split.
8. **Map** the samples back to the original columns and values.
9. **Evaluate** against the complete train and test splits: Shape, Trend, α-Precision, β-Recall and Utility (downstream task performance), following [TabSyn](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e9750610639c3e7a849cff746bf60dbd-Abstract-Conference.html) (ICLR 2024).

<details>
<summary>Repository structure</summary>

```
augmask/
├── augment.py           # step 5: LGBM-S (Algorithm 1), mean/mode fill, the Augmenter interface
├── run.py               # one method on one scenario: build, train, sample, evaluate
├── evaluate.py          # step 9: Shape, Trend, α-Precision, β-Recall, Utility
├── data/
│   ├── datasets.py      # step 1: download and clean the six UCI tables
│   ├── split.py         # step 2: fixed stratified 60/20/20 split
│   ├── missingness.py   # step 3: MCAR and MAR masks
│   ├── transforms.py    # steps 4 and 6: quantile-normal, integer codes, standardization
│   ├── scenario.py      # steps 3-6 and 8 for one scenario; saves and loads masks and fills
│   └── schema.py        # column types, target and task
├── models/              # step 7
│   ├── tabddpm.py, cdtd.py, tabdiff.py   # the backbones, each with an optional loss mask
│   └── common.py        # seeding, batching (one fill per step), EMA
└── baselines/
    ├── forestdiff.py    # the official ForestDiffusion package, wrapped
    └── miwae.py
scripts/
├── run_all.sh           # all experiments, then the summary tables
├── make_tables.py       # rank tables with Wilcoxon markers
└── plot_overview.py     # the figure above
tests/                   # pipeline, missingness, baselines, evaluation, each backbone, the runner
```

</details>

## 📝 Notes on this release

The data pipeline was rebuilt for this release (60/20/20 split, a new MAR generator), so numbers differ slightly from the paper. Most model and training settings follow Appendix C.6 of the paper. The exception is alternating fills: AugFull and AugMask alternate between 5 fills of the missing inputs during training, which performs better at high missing rates in this implementation (`--fills 1` restores the paper's single fill). LGBM-S follows Algorithm 1; `LGBMSAugmenter(log_var_offset=augmask.augment.LOG_CHI2_BIAS)` optionally corrects the downward bias of its variance model.

LGBM-S is a per-coordinate augmenter: it samples each missing entry independently given the observed values and the mask, so it does not model the dependence between entries that are missing together. An augmenter that samples the missing coordinates jointly is a natural direction for future work, and can be plugged in as described in [Your own augmenter](#your-own-augmenter).

## 📚 Citation

```bibtex
@inproceedings{kim2026augmask,
  title     = {{AugMask}: Training Diffusion Models on Incomplete Tabular Data via Stochastic Augmentation and Masking},
  author    = {Kim, Jungkyu and Park, Taeyoung and Lee, Kibok},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026}
}
```

## 🙏 Acknowledgements

The backbones are adapted from [TabDDPM](https://github.com/yandex-research/tab-ddpm), [CDTD](https://github.com/muellermarkus/cdtd) and [TabDiff](https://github.com/MinkaiXu/TabDiff) (MIT). The metrics follow [TabSyn](https://github.com/amazon-science/tabsyn) (Apache-2.0), α-Precision/β-Recall follow [synthcity](https://github.com/vanderschaarlab/synthcity), and MIWAE follows [hyperimpute](https://github.com/vanderschaarlab/hyperimpute) (MIT). ForestDiff uses the official [ForestDiffusion](https://github.com/SamsungSAILMontreal/ForestDiffusion) package.

Released under the MIT License.
