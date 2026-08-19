"""
prepare_arrow_dataset.py

Script to convert an image classification dataset into the
HuggingFace Arrow/Datasets format used throughout this codebase. Images are
processed lazily in shards to keep RAM usage bounded, the shards are then
compiled into a single dataset, the result is scanned for corrupt images,
and any corrupt samples are automatically repaired (dropped) at the end.

Images are resized so the SHORTER side equals --short_size (aspect ratio is
preserved), rather than warping to a fixed square.

Built-in dataset types (--dataset_type):
    imagenet     Standardized ILSVRC2012 layout (devkit-based train/val
                 collection). Supports --wnid_subset to restrict to a
                 predefined subset of classes (e.g. ImageNet-100).
    imagefolder  Any dataset laid out as <root>/<class>/<image>, i.e.
                 compatible with torchvision.datasets.ImageFolder. If
                 <root_dir>/<split> exists it is used as the split folder,
                 otherwise <root_dir> itself is treated as the split
                 (useful for datasets with no train/val distinction).

Extending to a new dataset: write a collector function
    def collect_my_dataset_samples(root_dir, split, wnid_filter=None):
        ...
        return samples, class_names
where `samples` is a list of (image_path, label) and `class_names[label]`
gives the human-readable/WNID name for that label, then register it in
DATASET_COLLECTORS below. `wnid_filter` can be ignored if it doesn't apply
to your dataset.

Usage:
    # ImageNet-100: create train split, shorter side = 256
    python prepare_arrow_dataset.py --dataset_type imagenet \
        --root_dir /path/to/ilsvrc2012 --output_dir /path/to/output/imgnet100_short256 \
        --split train --short_size 256 --wnid_subset in100

    # ImageNet-100: create val split, shorter side = 320
    python prepare_arrow_dataset.py --dataset_type imagenet \
        --root_dir /path/to/ilsvrc2012 \
        --output_dir /path/to/output/imgnet100_short320 \
        --split val \
        --short_size 320 \
        --wnid_subset in100

    # Any ImageFolder-compatible dataset (<root>/<class>/<image>)
    python prepare_arrow_dataset.py --dataset_type imagefolder \
        --root_dir /path/to/my_dataset --output_dir /path/to/output/my_dataset_short256 \
        --split train --short_size 256

    # Check a saved dataset
    python prepare_arrow_dataset.py \
        --output_dir /path/to/output/imgnet100_short256_train_single \
        --check_dataset
"""

import os
import io
import gc
import shutil
import time
import argparse
import psutil
import scipy.io as sio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data import DataLoader, Dataset

from datasets import (
    load_from_disk,
    concatenate_datasets,
    Dataset as HFDataset,
    Features,
    ClassLabel,
    Image as HFImage,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# ImageNet constants
# ---------------------------------------------------------------------------

EXTRACTED_FOLDERS = {
    "train": "ILSVRC2012_img_train",
    "val":   "ILSVRC2012_img_val",
    "devkit": "ILSVRC2012_devkit_t12",
}

IMAGENET_NUM_CLASSES = 1000

IMGNET100_WNIDS = [
    "n02869837", "n01749939", "n02488291", "n02107142", "n13037406",
    "n02091831", "n04517823", "n04589890", "n03062245", "n01773797",
    "n01735189", "n07831146", "n07753275", "n03085013", "n04485082",
    "n02105505", "n01983481", "n02788148", "n03530642", "n04435653",
    "n02086910", "n02859443", "n13040303", "n03594734", "n02085620",
    "n02099849", "n01558993", "n04493381", "n02109047", "n04111531",
    "n02877765", "n04429376", "n02009229", "n01978455", "n02106550",
    "n01820546", "n01692333", "n07714571", "n02974003", "n02114855",
    "n03785016", "n03764736", "n03775546", "n02087046", "n07836838",
    "n04099969", "n04592741", "n03891251", "n02701002", "n03379051",
    "n02259212", "n07715103", "n03947888", "n04026417", "n02326432",
    "n03637318", "n01980166", "n02113799", "n02086240", "n03903868",
    "n02483362", "n04127249", "n02089973", "n03017168", "n02093428",
    "n02804414", "n02396427", "n04418357", "n02172182", "n01729322",
    "n02113978", "n03787032", "n02089867", "n02119022", "n03777754",
    "n04238763", "n02231487", "n03032252", "n02138441", "n02104029",
    "n03837869", "n03494278", "n04136333", "n03794056", "n03492542",
    "n02018207", "n04067472", "n03930630", "n03584829", "n02123045",
    "n04229816", "n02100583", "n03642806", "n04336792", "n03259280",
    "n02116738", "n02108089", "n03424325", "n01855672", "n02090622",
]

WNID_PRESETS = {
    "in100": IMGNET100_WNIDS,
}


# ---------------------------------------------------------------------------
# ImageNet meta / val helpers
# ---------------------------------------------------------------------------

def parse_devkit_archive(root: Union[str, Path]):
    """Return (wnid_to_classes, val_wnids, idx_to_wnid) from the devkit."""

    def parse_meta_mat(devkit_root):
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
        nums_children = list(zip(*meta))[4]
        meta = [meta[i] for i, nc in enumerate(nums_children) if nc == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(c.split(", ")) for c in classes]
        idx_to_wnid   = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: c  for wnid, c  in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root):
        fpath = os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
        with open(fpath) as f:
            return [int(line) for line in f]

    devkit_root = os.path.join(root, EXTRACTED_FOLDERS["devkit"])
    idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
    val_idcs  = parse_val_groundtruth_txt(devkit_root)
    val_wnids = [idx_to_wnid[i] for i in val_idcs]
    return wnid_to_classes, val_wnids, idx_to_wnid


def filter_and_remap_samples(
    samples: List[Tuple[str, int]],
    sample_wnids: List[str],
    wnid_filter: List[str],
) -> List[Tuple[str, int]]:
    """Keep only samples whose wnid is in wnid_filter and remap labels to 0..N-1.

    Labels are assigned in sorted wnid order for determinism.
    """
    wnid_to_new_idx = {wnid: i for i, wnid in enumerate(sorted(wnid_filter))}
    filtered = [
        (path, wnid_to_new_idx[wnid])
        for (path, _), wnid in zip(samples, sample_wnids)
        if wnid in wnid_to_new_idx
    ]
    print(f"  Subset filter: kept {len(filtered)} / {len(samples)} samples "
          f"({len(wnid_filter)} classes)")
    return filtered


# ---------------------------------------------------------------------------
# Dataset collectors
#
# Each collector has the signature (root_dir, split, wnid_filter=None) and
# returns (samples, class_names), where samples is a list of
# (image_path, label) and class_names[label] is the name of that class.
# Register new collectors in DATASET_COLLECTORS at the bottom of this section.
# ---------------------------------------------------------------------------

def collect_imagenet_samples(
    root_dir: Union[str, Path],
    split: str,
    wnid_filter: Optional[List[str]] = None,
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Collect (path, label) pairs for the standardized ILSVRC2012 layout."""
    train_folder = os.path.join(root_dir, EXTRACTED_FOLDERS["train"])
    train_ds = ImageFolder(train_folder, loader=default_loader)
    train_map = train_ds.class_to_idx  # wnid -> idx

    if split == "train":
        samples = train_ds.samples
        sample_wnids = [train_ds.classes[idx] for _, idx in samples]
        print(f"Found {len(samples)} training images across {len(train_ds.classes)} classes")
    else:
        _, val_wnids, _ = parse_devkit_archive(root_dir)
        val_folder = os.path.join(root_dir, EXTRACTED_FOLDERS["val"])
        val_images = sorted(
            os.path.join(val_folder, f)
            for f in os.listdir(val_folder)
            if f.lower().endswith(('.jpeg', '.jpg', '.png'))
        )
        assert len(val_images) == len(val_wnids), (
            f"Mismatch: {len(val_images)} images vs {len(val_wnids)} ground-truth labels"
        )
        samples = [(path, train_map[wnid]) for path, wnid in zip(val_images, val_wnids)]
        sample_wnids = val_wnids
        print(f"Found {len(samples)} validation images")

    if wnid_filter is not None:
        samples = filter_and_remap_samples(samples, sample_wnids, wnid_filter)
        class_names = sorted(wnid_filter)
    else:
        class_names = train_ds.classes

    return samples, class_names


def collect_imagefolder_samples(
    root_dir: Union[str, Path],
    split: str,
    wnid_filter: Optional[List[str]] = None,
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Collect (path, label) pairs for any torchvision-ImageFolder-compatible
    dataset, i.e. a directory laid out as <root>/<class>/<image>.

    If <root_dir>/<split> exists it is used as the split folder (e.g. a
    dataset with separate train/ and val/ subfolders); otherwise root_dir
    itself is treated as the split (for datasets with no train/val split).
    """
    split_dir = os.path.join(root_dir, split)
    dataset_dir = split_dir if os.path.isdir(split_dir) else root_dir

    ds = ImageFolder(dataset_dir, loader=default_loader)
    print(f"Found {len(ds.samples)} images across {len(ds.classes)} classes in {dataset_dir}")

    if wnid_filter is not None:
        print("  Note: --wnid_subset is ignored for --dataset_type imagefolder")

    return ds.samples, ds.classes


DATASET_COLLECTORS = {
    "imagenet": collect_imagenet_samples,
    "imagefolder": collect_imagefolder_samples,
}


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def resize_shorter_side(image: Image.Image, short_size: int) -> Image.Image:
    """Resize so the shorter side == short_size, preserving aspect ratio."""
    w, h = image.size
    if w <= h:
        new_w = short_size
        new_h = int(round(h * short_size / w))
    else:
        new_h = short_size
        new_w = int(round(w * short_size / h))
    return image.resize((new_w, new_h), Image.Resampling.BILINEAR)


def load_and_process_image(
    path: str,
    label: int,
    short_size: int,
) -> Optional[Dict[str, Any]]:
    """Load, resize (shorter side), JPEG-compress, return dict or None on error."""
    try:
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if image.size[0] == 0 or image.size[1] == 0:
            return None

        image = resize_shorter_side(image, short_size)

        # JPEG-compress to strip metadata and keep file sizes manageable
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        image = Image.open(buf)

        return {"image": image, "label": label}

    except Exception as e:
        print(f"  Warning: skipping {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def get_memory_usage_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


# ---------------------------------------------------------------------------
# Core: lazy shard-based Arrow dataset creation
# ---------------------------------------------------------------------------

def create_arrow_dataset_lazy(
    samples: List[Tuple[str, int]],
    output_path: str,
    split_name: str,
    short_size: int,
    batch_size: int = 2048,
    available_ram_gb: float = 28.0,
    max_shard_size: str = "200MB",
    num_classes: int = IMAGENET_NUM_CLASSES,
):
    """
    Process `samples` in batches, writing each batch as a shard to output_path.

    After this call, output_path contains a directory of shard sub-directories.
    Use `compile_shards()` to merge them into a single dataset.
    """
    os.makedirs(output_path, exist_ok=True)
    num_samples  = len(samples)
    total_batches = (num_samples + batch_size - 1) // batch_size
    print(f"\n--- Creating Arrow shards for '{split_name}' ---")
    print(f"  Samples   : {num_samples}")
    print(f"  Batch size: {batch_size}  ({total_batches} batches)")
    print(f"  Short side: {short_size}px  (aspect ratio preserved)")
    print(f"  Output    : {output_path}")

    features = Features({
        'image': HFImage(decode=True),
        'label': ClassLabel(num_classes=num_classes),
    })

    total_written = 0
    total_skipped = 0
    t0 = time.time()

    outer_bar = tqdm(
        range(0, num_samples, batch_size),
        desc=f"[Phase 1/3] Sharding {split_name}",
        total=total_batches,
        unit="shard",
    )
    for shard_idx, batch_start in enumerate(outer_bar):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_samples = samples[batch_start:batch_end]

        batch_data = []
        shard_skipped = 0
        for path, label in tqdm(
            batch_samples,
            desc=f"  shard {shard_idx+1:>{len(str(total_batches))}}/{total_batches} loading",
            leave=False,
            unit="img",
        ):
            result = load_and_process_image(path, label, short_size)
            if result is not None:
                batch_data.append(result)
            else:
                shard_skipped += 1

        if not batch_data:
            tqdm.write(f"  Shard {shard_idx+1}: no valid images, skipping entirely")
            continue

        batch_hf = HFDataset.from_list(batch_data).cast(features)
        shard_path = os.path.join(
            output_path,
            f"data-{shard_idx:05d}-of-{total_batches:05d}.arrow"
        )
        batch_hf.save_to_disk(shard_path, max_shard_size=max_shard_size)

        total_written += len(batch_data)
        total_skipped += shard_skipped
        if shard_skipped:
            tqdm.write(f"  Shard {shard_idx+1}: wrote {len(batch_data)}, skipped {shard_skipped}")

        outer_bar.set_postfix(
            written=total_written,
            skipped=total_skipped,
            ram_gb=f"{get_memory_usage_gb():.1f}",
        )

        del batch_data, batch_hf
        gc.collect()

        if get_memory_usage_gb() > available_ram_gb * 0.9:
            tqdm.write(f"  Warning: RAM at {get_memory_usage_gb():.1f}/{available_ram_gb:.1f} GB, forcing GC")
            gc.collect()

    elapsed = time.time() - t0
    print(f"\n  Phase 1 done in {elapsed:.0f}s — "
          f"wrote {total_written:,} images, skipped {total_skipped:,}, "
          f"across {total_batches} shards → {output_path}")


def compile_shards(shards_dir: str, max_shard_size: str = "200MB") -> str:
    """Concatenate all shard sub-directories into a single dataset next to shards_dir."""
    target_path = shards_dir.rstrip("/") + "_single"
    print(f"\n[Phase 2/3] Compiling shards into single dataset")
    print(f"  Source : {shards_dir}")
    print(f"  Target : {target_path}")

    shard_dirs = sorted(d for d in Path(shards_dir).iterdir() if d.is_dir())
    print(f"  Merging {len(shard_dirs)} shard directories...")
    t0 = time.time()
    datasets = [load_from_disk(str(d)) for d in tqdm(shard_dirs, desc="  Loading shards", unit="shard")]
    combined = concatenate_datasets(datasets)

    os.makedirs(target_path, exist_ok=True)
    print(f"  Saving {len(combined):,} samples (this may take a moment)...")
    combined.save_to_disk(target_path, max_shard_size=max_shard_size)
    print(f"  Phase 2 done in {time.time()-t0:.0f}s — {len(combined):,} samples → {target_path}")
    return target_path


# ---------------------------------------------------------------------------
# Corruption scan + repair
# ---------------------------------------------------------------------------

def scan_for_corrupt(dataset_path: str) -> List[int]:
    """Iterate through every sample in the saved dataset and force-decode each image.

    Returns a list of indices that could not be loaded (corrupt / truncated).
    Prints a summary at the end.
    """
    print(f"\n[Phase 3/3] Full corruption scan: {dataset_path}")
    hf_ds = load_from_disk(dataset_path)
    n = len(hf_ds)
    print(f"  Scanning {n:,} samples ...")
    corrupt_indices = []
    t0 = time.time()
    bar = tqdm(range(n), desc="  Scanning", unit="img")
    for i in bar:
        try:
            sample = hf_ds[i]
            image = sample['image']
            if image is None:
                tqdm.write(f"  Index {i}: None image object")
                corrupt_indices.append(i)
                continue
            image.load()  # force full pixel decode
        except Exception as e:
            tqdm.write(f"  Index {i}: {e}")
            corrupt_indices.append(i)
        if corrupt_indices:
            bar.set_postfix(corrupt=len(corrupt_indices))
    if corrupt_indices:
        print(f"  Phase 3 done in {time.time()-t0:.0f}s — "
              f"found {len(corrupt_indices)} corrupt sample(s): {corrupt_indices[:20]}"
              f"{'...' if len(corrupt_indices) > 20 else ''}")
    else:
        print(f"  Phase 3 done in {time.time()-t0:.0f}s — all {n:,} samples OK.")
    return corrupt_indices


def repair_dataset(input_path: str, output_path: str, num_proc: int):
    """
    Repairs a corrupt Arrow dataset by first disabling automatic image decoding,
    then safely filtering out corrupt samples, and finally re-enabling decoding
    for the clean, saved dataset.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input directory not found: {input_path}")
        return

    if os.path.exists(output_path):
        print(f"Warning: Output directory {output_path} already exists. It will be removed.")
        shutil.rmtree(output_path)

    print(f"--- Loading dataset from: {input_path} ---")
    corrupt_dataset = load_from_disk(input_path)
    total_samples = len(corrupt_dataset)
    print(f"Found {total_samples:,} total samples.")

    # Disable automatic decoding so a corrupt sample doesn't crash iteration:
    # `example['image']` becomes a raw {'bytes': ...} dict instead of a PIL Image.
    print("Disabling automatic image decoding to take control...")
    dataset_no_decode = corrupt_dataset.cast_column('image', HFImage(decode=False))

    def is_sample_valid(example):
        image_dict = example['image']
        if not image_dict or 'bytes' not in image_dict or not image_dict['bytes']:
            return False
        try:
            with Image.open(io.BytesIO(image_dict['bytes'])) as img:
                img.load()  # force reading the data
            return True
        except Exception:
            return False

    print(f"--- Filtering out corrupt samples using {num_proc} process(es)... ---")
    clean_dataset_no_decode = dataset_no_decode.filter(
        is_sample_valid,
        num_proc=num_proc,
        desc="Validating samples"
    )

    valid_samples = len(clean_dataset_no_decode)
    corrupt_count = total_samples - valid_samples

    print("\n--- Filtering Complete ---")
    print(f"Original samples: {total_samples:,}")
    print(f"Valid samples found: {valid_samples:,}")
    print(f"Corrupt samples removed: {corrupt_count:,}")

    print("Re-enabling automatic image decoding for the clean dataset...")
    final_dataset = clean_dataset_no_decode.cast_column('image', HFImage(decode=True))

    print(f"\n--- Saving clean dataset to: {output_path} ---")
    final_dataset.save_to_disk(output_path, num_proc=num_proc)
    print("--- Repair process finished successfully! ---")


# ---------------------------------------------------------------------------
# Dataset wrapper for checking
# ---------------------------------------------------------------------------

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform  = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image  = sample['image']
        if self.transform:
            image = self.transform(image)
        return image, sample['label']


def check_dataset(dataset_path: str, batch_size: int = 32):
    """Load a saved dataset and print basic statistics for one batch."""
    print(f"\n--- Checking dataset at {dataset_path} ---")
    hf_ds = load_from_disk(dataset_path)
    print(f"  Total samples : {len(hf_ds)}")
    print(f"  Features      : {hf_ds.features}")

    # Check a single raw sample to see the stored image size
    raw = hf_ds[0]
    img0 = raw['image']
    print(f"  First image size (W x H): {img0.size}  mode: {img0.mode}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def pad_collate(batch):
        """Pad images to the max H and W in the batch so they can be stacked."""
        images, labels = zip(*batch)
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        padded = []
        for img in images:
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            # F.pad order: (left, right, top, bottom)
            padded.append(torch.nn.functional.pad(img, (0, pad_w, 0, pad_h)))
        return torch.stack(padded), torch.tensor(labels)

    loader = DataLoader(
        HFDatasetWrapper(hf_ds, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=pad_collate,
    )
    images, labels = next(iter(loader))
    print(f"  Batch images shape : {images.shape}")
    print(f"  Batch labels shape : {labels.shape}")
    print(f"  Label range        : [{labels.min().item()}, {labels.max().item()}]")
    print(f"  Pixel range        : [{images.min():.3f}, {images.max():.3f}]")

    # Save a grid for visual confirmation
    grid = torchvision.utils.make_grid(
        torch.clamp(images[:16] * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                    + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1), 0, 1),
        nrow=8, padding=2
    )
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("check_batch.png", bbox_inches="tight")
    print("  Saved visual grid to check_batch.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an Arrow/HuggingFace image classification dataset with shorter-side resizing."
    )
    parser.add_argument('--dataset_type', type=str, default='imagenet',
                        choices=list(DATASET_COLLECTORS.keys()),
                        help='Which dataset layout to collect samples from (default: imagenet)')
    parser.add_argument('--root_dir',    type=str, default=None,
                        help='Path to the raw dataset root (for imagenet: contains ILSVRC2012_img_train etc.; '
                             'for imagefolder: contains one subfolder per class, optionally under a split subdir)')
    parser.add_argument('--output_dir',  type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--split',       type=str, default='train',
                        choices=['train', 'val', 'both'],
                        help='Which split(s) to process')
    parser.add_argument('--short_size',  type=int, default=256,
                        help='Resize shorter side to this many pixels (default: 256)')
    parser.add_argument('--batch_size',  type=int, default=2048,
                        help='Images per shard batch (default: 2048)')
    parser.add_argument('--ram_gb',      type=float, default=28.0,
                        help='Available RAM in GB for memory monitoring (default: 28)')
    parser.add_argument('--max_shard_size', type=str, default="200MB",
                        help='Max size per Arrow shard file (default: 200MB)')
    parser.add_argument('--wnid_subset',  type=str, default=None,
                        choices=list(WNID_PRESETS.keys()),
                        help='(imagenet only) Filter to a predefined WNID subset, e.g. in100')
    parser.add_argument('--check_dataset', action='store_true',
                        help='Skip creation; just load and check --output_dir')
    parser.add_argument('--num_proc', type=int, default=os.cpu_count(),
                        help='Number of processes for repair filtering (default: all CPUs)')
    args = parser.parse_args()

    if args.check_dataset:
        check_dataset(args.output_dir)

    else:
        assert args.root_dir is not None, "--root_dir is required for dataset creation"

        wnid_filter = WNID_PRESETS[args.wnid_subset] if args.wnid_subset else None
        collect_fn = DATASET_COLLECTORS[args.dataset_type]

        splits_to_run = ['train', 'val'] if args.split == 'both' else [args.split]

        print(f"\n{'='*60}")
        print(f"  Arrow dataset creation")
        print(f"  Dataset type: {args.dataset_type}")
        print(f"  Split(s)    : {', '.join(splits_to_run)}")
        print(f"  Short side  : {args.short_size}px  (aspect ratio preserved)")
        print(f"  Batch size  : {args.batch_size} imgs/shard")
        print(f"  Output      : {args.output_dir}")
        print(f"  Phases      : [1] shard  →  [2] compile  →  [3] corruption scan")
        print(f"{'='*60}")

        for split in splits_to_run:
            print(f"\n{'─'*60}")
            print(f"  Split: {split}")
            print(f"{'─'*60}")

            samples, class_names = collect_fn(args.root_dir, split, wnid_filter)
            num_classes = len(class_names)

            shards_dir = os.path.join(args.output_dir, split)

            # Phase 1: write shards
            create_arrow_dataset_lazy(
                samples       = samples,
                output_path   = shards_dir,
                split_name    = split,
                short_size    = args.short_size,
                batch_size    = args.batch_size,
                available_ram_gb = args.ram_gb,
                max_shard_size   = args.max_shard_size,
                num_classes      = num_classes,
            )

            # Phase 2: merge shards into single dataset
            final_path = compile_shards(shards_dir, max_shard_size=args.max_shard_size)
            print(f"\nFinal dataset for '{split}' saved at: {final_path}")

            # Phase 3: full corruption scan + auto-repair
            corrupt_indices = scan_for_corrupt(final_path)
            if corrupt_indices:
                repaired_path = final_path + "_repaired"
                print(f"\n  Auto-repairing {len(corrupt_indices)} corrupt sample(s) → {repaired_path}")
                repair_dataset(final_path, repaired_path, num_proc=args.num_proc)
                print(f"  Clean dataset saved at: {repaired_path}")
