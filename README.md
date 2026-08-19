# Co-observation

Code for the paper:

> **[Forgetting, plasticity, and co-observation: a third facet of continual learning]**
> [Timm Hess, Abhishek Jha, Gido M. van de Ven, Tinne Tuytelaar]
> [CoLLAs, 2026]
<!-- > [arXiv / DOI link] -->

This repository contains the training and evaluation code for continual learning experiments across a range of scenarios, backbones, and CL strategies. It is built on top of a vendored/patched version of [Avalanche](https://github.com/ContinualAI/avalanche).



## Setup

```bash
pip install -r requirements.txt
```

The `avalanche/` directory is a vendored and patched version of Avalanche.

For SSL training, you'll also need our fork of [CaSSLE](https://github.com/TimmHess/cassle) — see [Training SSL backbones (CaSSLE)](#training-ssl-backbones-cassle) below.




## Quick start (CIFAR-100, no data prep needed)


```bash
python train.py \
    --config_base experiment_scripts \
    --config_path dataset/cifar100 scenario/chunked_4x25 model/slim_resnet18_BN \
                   method/finetune train_regime/eps90_bs256 train_regime/crop_flip \
                   optimizer/OneCycleSGD_lr01_wd \
                   evaluation/downstream_probing_sgd evaluation/downstream_cifar100 \
    --skip_initial_eval \
    --eval_exps 0 \
    --bs 128 \
    --overwrite_input_size 32 32 \
    --iterations_per_task 9750 \
    --exp_name naive \
    --seed 142
```

This runs a CIFAR-100 experiment (4 chunks) with a slim ResNet-18 backbone, naive continual learning, and downstream linear-probe evaluation. No `--dset_rootpath` is needed here: it defaults to `./data`, and CIFAR-100 is downloaded there automatically if not already present (see [Dataset root paths in YAML files](#dataset-root-paths-in-yaml-files)).



## Config system

Many hyperparameters can be set via composable YAML snippets. Pass a base config directory and a list of relative YAML paths:

```bash
python train.py \
    --config_base experiment_scripts \
    --config_path dataset/cifar100 scenario/chunked_4x25 model/resnet18 method/finetune
```

Each path is resolved as `<config_base>/<config_path>.yaml`. YAML values are loaded in order, later files override earlier ones. Any argument explicitly passed on the command line takes priority over YAML values.

The config snippets under `experiment_scripts/` are organised by role:

| Directory      | Controls |
|----------------|----------|
| `dataset/`     | Scenario, number of experiences, dataset root path |
| `scenario/`    | Number of experiences, class ordering |
| `model/`       | Backbone architecture |
| `method/`      | CL strategy, replay buffer size |
| `optimizer/`   | Optimiser, learning rate, schedule |
| `train_regime/`| Epochs, batch size, augmentation |
| `evaluation/`  | Linear probing, downstream evaluation sets |


### Dataset root paths in YAML files

`dataset/*.yaml` configs leave `dset_rootpath` commented out by default. For datasets that torchvision/Avalanche can auto-download (CIFAR-100, MNIST, ...), that's all you need — they're fetched to `--dset_rootpath`'s default (`./data`).

For datasets that require manual preparation (e.g. the Arrow-format ImageNet variants, see [below](#imagenet-scale-datasets-arrow-format)), either:
- pass `--dset_rootpath /path/to/your/data` on the command line (overrides any YAML value), or
- uncomment and edit the `dset_rootpath:` line in the relevant `dataset/*.yaml` file.

The same applies to `downstream_rootpaths` in `evaluation/downstream_*.yaml` configs: `null` relies on automatic download, otherwise it must point at your local copy of that dataset.




## Training SSL backbones (CaSSLE)

SSL methods (`BarlowTwins` and `I-JEPA`) are **evaluated** with this repository, but **trained** using our fork of [CaSSLE](https://github.com/TimmHess/cassle), adapted for continual self-supervised pretraining.

### 1. Train in cassle (SLURM example)

From a `cassle` checkout, submit a continual SSL run. `exp_coob/continual/IN100/split4x25/barlow_split4.sh` trains BarlowTwins on a 4-task class-incremental split of ImageNet-100. `main_continual.py` loops over all tasks internally producing a checkpoint sequence:

```bash
sbatch --job-name=barlow_FT_seed142 slurm_run.sh \
    "SEED=142 bash exp_coob/continual/IN100/split4x25/barlow_split4.sh"
```

This writes one checkpoint per task under `$EXP_TARGET_DIR/$NAME/checkpoints/` (here, `.../cassle/IN100/split_4x25/barlow/barlow_FT_seed142/checkpoints/`):

```
barlow_FT_seed142-task_0.ckpt
barlow_FT_seed142-task_1.ckpt
barlow_FT_seed142-task_2.ckpt
barlow_FT_seed142-task_3.ckpt
```

### 2. Evaluate the checkpoints

Point `--eval_stored_weights` at that checkpoint directory. `train.py` loads one checkpoint per experience and remaps cassle's `encoder.*` state-dict keys to this codebase's `feature_extractor.*` via `--weights_replace_key_dict cassle`:

```bash
python train.py \
    --config_base experiment_scripts \
    --config_path dataset/IN100 scenario/split_4x25 model/resnet18 \
                   method/barlow_twins_cassle \
                   train_regime/100k_itr_bs_256 train_regime/simclr \
                   optimizer/sgd \
                   evaluation/downstream_probing_sgd evaluation/downstream_IN100_V2_eval \
    --dset_rootpath /path/to/your/imgnet100_short256 \
    --use_amp \
    --skip_initial_eval \
    --bs 256 \
    --weight_decay 0.05 \
    --iterations_per_task 0 \
    --eval_exps 0 \
    --store_models False \
    --weights_replace_key_dict cassle \
    --eval_stored_weights /path/to/cassle/IN100/split_4x25/barlow/barlow_FT_seed142/checkpoints/ \
    --exp_name eval_DS_SGD_ft \
    --seed 142
```

Use `--weights_replace_key_dict cassle_ijepa` instead when evaluating I-JEPA checkpoints.


## Preparing datasets (Arrow format)

Large image datasets (e.g. ImageNet-100) are pre-processed once into the HuggingFace Arrow/Datasets format via `prepare_arrow_dataset.py`. It processes images lazily in shards to keep RAM usage bounded, compiles the shards into a single dataset, scans for corrupt images, and auto-repairs (drops) any it finds.

It ships with two dataset layouts out of the box, selected via `--dataset_type`:
- `imagenet` — the standardized ILSVRC2012 layout, with `--wnid_subset` to filter to a predefined subset of classes (currently `in100` for ImageNet-100).
- `imagefolder` — any dataset laid out as `<root>/<class>/<image>` (i.e. compatible with `torchvision.datasets.ImageFolder`), for datasets not covered by the paper.

To support a dataset that doesn't fit either layout, add a collector function `(root_dir, split, wnid_filter=None) -> (samples, class_names)` and register it in `DATASET_COLLECTORS` at the top of the script — the shard/compile/scan/repair pipeline is dataset-agnostic and needs no changes.

```bash
# Prepare training split (shorter side resized to 256 px, aspect ratio preserved)
python prepare_arrow_dataset.py --dataset_type imagenet \
  --root_dir /path/to/ILSVRC2012 \
  --output_dir /path/to/output/imgnet100_short256 \
  --split train \
  --short_size 256 \
  --wnid_subset in100

# Prepare validation split
python prepare_arrow_dataset.py --dataset_type imagenet \
  --root_dir /path/to/ILSVRC2012 \
  --output_dir /path/to/output/imgnet100_short256 \
  --split val \
  --short_size 256 \
  --wnid_subset in100

# Verify the saved dataset
python prepare_arrow_dataset.py \
  --output_dir /path/to/output/imgnet100_short256 \
  --check_dataset
```

Point `--dset_rootpath` at the output directory when running `train.py`.



## Outputs

Results are saved under `--save_path` in a subdirectory named `{exp_name}{exp_postfix}`. The postfix defaults to a timestamp + UID, ensuring unique directories per run. Set `--exp_postfix ""` for a fixed directory (useful for checkpointing/resuming).

Model checkpoints are saved after each experience by default (`--store_models True`).



## Reproducing the main experiments

One example call per dataset / training regime from the paper's main results. The `--seed` shown is one representative seed; the paper reports means over multiple seeds.

All examples below use ImageNet-100, which is not auto-downloaded — prepare it first (see [Preparing datasets (Arrow format)](#preparing-datasets-arrow-format)) and pass `--dset_rootpath /path/to/your/imgnet100_short256` to each command (see [Dataset root paths in YAML files](#dataset-root-paths-in-yaml-files)).

### ImageNet-100 — Naive

```bash
python train.py \
    --config_base experiment_scripts \
    --config_path dataset/IN100 scenario/chunked_4x25 model/resnet18 \
                   method/finetune train_regime/eps90_bs256 train_regime/randaug \
                   optimizer/OneCycleSGD_lr01_wd \
                   evaluation/downstream_probing evaluation/downstream_IN100 \
    --dset_rootpath /path/to/your/imgnet100_short256 \
    --use_amp \
    --skip_initial_eval \
    --eval_exps 0 \
    --iterations_per_task 11070 \
    --exp_name finetune_OptimParam \
    --seed 152
```

### ImageNet-100 — Incremental Joint

```bash
python train.py \
    --config_base experiment_scripts \
    --config_path dataset/IN100 scenario/chunked_4x25 model/resnet18 \
                   method/joint train_regime/eps90_bs256 train_regime/randaug \
                   optimizer/OneCycleSGD_lr01_wd \
                   evaluation/downstream_probing evaluation/downstream_IN100 \
    --dset_rootpath /path/to/your/imgnet100_short256 \
    --use_amp \
    --skip_initial_eval \
    --eval_exps 0 \
    --iterations_per_task 11070 \
    --exp_name joint_incr_OptimParam \
    --seed 152
```

### ImageNet-100 — Joint (offline)

```bash
python train.py \
    --config_base experiment_scripts \
    --config_path dataset/IN100 scenario/repeat_dataset model/resnet18 \
                   method/finetune train_regime/eps90_bs256 train_regime/randaug \
                   optimizer/OneCycleSGD_lr01_wd \
                   evaluation/downstream_probing_sgd evaluation/downstream_IN100_V2_eval \
    --dset_rootpath /path/to/your/imgnet100_short256 \
    --use_amp \
    --skip_initial_eval \
    --eval_exps 0 \
    --iterations_per_task 44280 \
    --num_experiences 1 \
    --exp_name joint_upper_OptimParam \
    --seed 142
```

### ImageNet-100 — LwF

```bash
python train.py \
    --config_base experiment_scripts \
    --config_path dataset/IN100 scenario/chunked_4x25 model/resnet18 \
                   method/lwf train_regime/eps90_bs256 train_regime/randaug \
                   optimizer/OneCycleSGD_lr01_wd \
                   evaluation/downstream_probing evaluation/downstream_IN100 \
    --dset_rootpath /path/to/your/imgnet100_short256 \
    --task_incr \
    --use_amp \
    --skip_initial_eval \
    --eval_exps 0 \
    --iterations_per_task 11070 \
    --exp_name lwf_mh_OptimParam \
    --seed 142
```

### ImageNet-100 — Replay

Buffer size (per chunk) is set via `method/replay_cumu_<size>_EB` — available sizes: `5000`, `25000`, `50000`. Swap the config path and `--exp_name` accordingly for each buffer size.

```bash
python train.py \
    --config_base experiment_scripts \
    --config_path dataset/IN100 scenario/chunked_4x25 model/resnet18 \
                   method/replay_cumu_5000_EB train_regime/eps90_bs256 train_regime/randaug \
                   optimizer/OneCycleSGD_lr01_wd \
                   evaluation/downstream_probing evaluation/downstream_IN100 \
    --dset_rootpath /path/to/your/imgnet100_short256 \
    --use_amp \
    --skip_initial_eval \
    --eval_exps 0 \
    --iterations_per_task 11070 \
    --exp_name replay_cumu_5000_EB_OptimParam \
    --seed 142
```

### ImageNet-100 — Ensemble

Trained the same way as [Naive](#imagenet-100--naive) above (`method/ensemble` sets `strategy: finetune`, same as Naive) — only the evaluation differs, using `use_ensemble: True` and a concatenated-linear-probe downstream metric across the per-experience checkpoints. Point `--eval_stored_weights` at the `model_weights/` subdirectory of that Naive run's own output (`<save_path>/<exp_name><exp_postfix>/model_weights/`, see [Outputs](#outputs)):

```bash
python train.py \
    --config_base experiment_scripts \
    --config_path dataset/IN100 scenario/chunked_4x25 model/resnet18 \
                   method/ensemble train_regime/eps90_bs256 train_regime/randaug \
                   optimizer/sgd \
                   evaluation/downstream_probing_sgd evaluation/downstream_IN100_V2_eval \
    --dset_rootpath /path/to/your/imgnet100_short256 \
    --use_amp \
    --skip_initial_eval \
    --eval_exps 0 \
    --iterations_per_task 0 \
    --downstream_method concat_linear \
    --store_models False \
    --eval_stored_weights /path/to/naive_run_results/finetune_OptimParam.../model_weights/ \
    --exp_name eval_DS_SGD_ens_OptimParam \
    --seed 142
```

### ImageNet-100 — BarlowTwins (SSL, Finetune)

Trained in cassle, evaluated here — see [Training SSL backbones (CaSSLE)](#training-ssl-backbones-cassle) for the full mechanism.

```bash
# 1. Train (cassle)
sbatch --job-name=barlow_FT_seed142 slurm_run.sh \
    "SEED=142 bash exp_coob/continual/IN100/split4x25/barlow_split4.sh"

# 2. Evaluate (Coob_eval)
python train.py \
    --config_base experiment_scripts \
    --config_path dataset/IN100 scenario/split_4x25 model/resnet18 \
                   method/barlow_twins_cassle \
                   train_regime/100k_itr_bs_256 train_regime/simclr \
                   optimizer/sgd \
                   evaluation/downstream_probing_sgd evaluation/downstream_IN100_V2_eval \
    --dset_rootpath /path/to/your/imgnet100_short256 \
    --use_amp \
    --skip_initial_eval \
    --bs 256 \
    --weight_decay 0.05 \
    --iterations_per_task 0 \
    --eval_exps 0 \
    --store_models False \
    --weights_replace_key_dict cassle \
    --eval_stored_weights /path/to/cassle/IN100/split_4x25/barlow/barlow_FT_seed142/checkpoints/ \
    --exp_name eval_DS_SGD_ft \
    --seed 142
```

### ImageNet-100 — I-JEPA (SSL, Finetune)

```bash
# 1. Train (cassle)
sbatch --job-name=ijepa_ft_sd142 slurm_run.sh \
    "SEED=142 bash exp_coob/continual/IN100/ijepa_chunked4x.sh"

# 2. Evaluate (Coob_eval)
python train.py \
    --config_base experiment_scripts \
    --config_path dataset/IN100 scenario/chunked_4x25 model/vit_base_jepa \
                   method/ijepa \
                   train_regime/100k_itr_bs_256 train_regime/simclr \
                   optimizer/sgd \
                   evaluation/downstream_probing_sgd evaluation/downstream_IN100_V2_eval \
    --dset_rootpath /path/to/your/imgnet100_short256 \
    --use_amp \
    --skip_initial_eval \
    --bs 256 \
    --weight_decay 0.05 \
    --iterations_per_task 0 \
    --eval_exps 0 \
    --store_models False \
    --weights_replace_key_dict cassle_ijepa \
    --eval_stored_weights /path/to/cassle/IN100_V2/chunked_4x/ijepa/ijepa_ft_sd142/checkpoints/ \
    --exp_name eval_DS_SGD_ft \
    --seed 142
```

