# Deterministic Reversible Data Augmentation for Neural Machine Translation

The relevant code for the Findings of ACL 2024 long paper [Deterministic Reversible Data Augmentation for Neural Machine Translation](https://arxiv.org/abs/2406.02517).

## Preparation

Download iwslt14 data [here](https://github.com/facebookresearch/fairseq/blob/main/examples/translation/prepare-iwslt14.sh).

Install the [fairseq v0.12.2](https://github.com/facebookresearch/fairseq/releases/tag/v0.12.2) library, and then copy or overwrite the adapted files in the `fairseq_drda` folder to the corresponding directory.

Arrange your workspace as outlined below.

```raw
DRDA/
│
└─── data/
    │
    └─── iwslt14/
        │
        └─── de-en/
            │
            └─── train.de-en.de
            └─── train.de-en.en
            └─── valid.de-en.en
            └─── valid.de-en.en
            └─── test.de-en.en
            └─── test.de-en.en
│
└─── fairseq/
│
└─── scripts/
```

## Preprocessing, Training, and Evaluation

Use `scripts/preprocess.sh` to apply deterministic multi-granularity segmentation, and use `scripts/binairze.sh` to create binary files for faster training.

```bash
# usage: bash preprocess.sh [expt_name]

bash scripts/preprocess.sh de-en_5k_10k
bash scripts/binairze.sh
```

Set relevant hyper-paraments and start training.

```bash
bash scripts/train.sh
```

After training, evaluate the translation model.
```bash
# usage: bash eval.sh [expt_name] [split] [granularity]

bash eval.sh de-en_5k_10k test 10000
```