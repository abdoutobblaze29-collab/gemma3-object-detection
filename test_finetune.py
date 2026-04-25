import os
import argparse
import logging
from functools import partial

import torch
import albumentations as A

from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from config import Configuration
from utils import (
    test_collate_function,
    visualize_bounding_boxes,
)

os.makedirs("outputs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def get_augmentations(cfg):
    resize_size = 512 if "SmolVLM" in cfg.model_id else 896

    return A.Compose(
        [
            A.Resize(height=resize_size, width=resize_size),
        ]
    )

from datasets import load_dataset

def debug_dataset_labels(cfg, split="test", n=10):
    print("\n" + "=" * 50)
    print(f"DEBUG DATASET LABELS: {cfg.dataset_id} [{split}]")
    print("=" * 50)

    ds = load_dataset(
        cfg.dataset_id,
        split=split,
        download_mode="force_redownload",
    )

    for i in range(min(n, len(ds))):
        print(f"\nSample {i}")
        print(ds[i]["label_for_paligemma"])

    print("=" * 50 + "\n")

def get_existing_split(dataset_id, preferred_split="test"):
    dataset_dict = load_dataset(dataset_id)

    if preferred_split in dataset_dict:
        return preferred_split

    if preferred_split == "test" and "validation" in dataset_dict:
        return "validation"

    if preferred_split == "test" and "valid" in dataset_dict:
        return "valid"

    if preferred_split == "validation" and "valid" in dataset_dict:
        return "valid"

    if preferred_split == "valid" and "validation" in dataset_dict:
        return "validation"

    raise ValueError(
        f"Could not find split '{preferred_split}' in dataset {dataset_id}. "
        f"Available splits: {list(dataset_dict.keys())}"
    )


def print_cfg(cfg):
    print("\n" + "=" * 50)
    print("CONFIGURATION")
    print("=" * 50)

    for key, value in vars(cfg).items():
        print(f"{key:20s}: {value}")

    print("=" * 50 + "\n")

def get_dataloader(processor, cfg, split="test"):
    actual_split = get_existing_split(cfg.dataset_id, preferred_split=split)

    logger.info(f"Loading dataset split: {actual_split}")

    test_dataset = load_dataset(cfg.dataset_id, split=actual_split)

    test_collate_fn = partial(
        test_collate_function,
        processor=processor,
        device=cfg.device,
        transform=get_augmentations(cfg),
    )

    return DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=test_collate_fn,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 0),

        # test_collate_function already moves tensors to cfg.device.
        # pin_memory=True crashes if the batch already contains CUDA tensors.
        pin_memory=False,
    )


def predict_batch(model, processor, sample, max_new_tokens=128):
    sample = sample.to(model.device)

    input_len = sample["input_ids"].shape[-1]

    with torch.no_grad():
        generated_ids = model.generate(
            **sample,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = generated_ids[:, input_len:]

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return decoded


def main():
    cfg = Configuration()

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_id", type=str, default=None)
    parser.add_argument("--checkpoint_id", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--attn_imp", type=str, default=None)

    args = parser.parse_args()

    if args.dataset_id:
        cfg.dataset_id = args.dataset_id

    if args.checkpoint_id:
        cfg.checkpoint_id = args.checkpoint_id

    if args.model_id:
        cfg.model_id = args.model_id

    if args.batch_size:
        cfg.batch_size = args.batch_size

    if args.attn_imp:
        cfg.attn_implementation = args.attn_imp

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Dataset: {cfg.dataset_id}")
    logger.info(f"Checkpoint: {cfg.checkpoint_id}")

    processor = AutoProcessor.from_pretrained(cfg.checkpoint_id)

    model_kwargs = {
        "torch_dtype": cfg.dtype,
        "device_map": "auto",
    }

    print_cfg(cfg)

    if hasattr(cfg, "attn_implementation"):
        model_kwargs["attn_implementation"] = cfg.attn_implementation

    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.checkpoint_id,
        **model_kwargs,
    )

    model.eval()

    test_dataloader = get_dataloader(
        processor=processor,
        cfg=cfg,
        split=args.split,
    )

    debug_dataset_labels(cfg, split="test", n=50)

    file_count = 0

    progress_bar = tqdm(test_dataloader, desc="Predicting")

    for sample, sample_images in progress_bar:
        predictions = predict_batch(
            model=model,
            processor=processor,
            sample=sample,
            max_new_tokens=args.max_new_tokens,
        )

        for output_text, sample_image in zip(predictions, sample_images):
            image = sample_image[0]

            width, height = image.size

            output_path = os.path.join(
                args.output_dir,
                f"output_{file_count}.png",
            )

            print(f"\nImage {file_count}")
            print("Prediction:", output_text)
            print("Saved:", output_path)

            visualize_bounding_boxes(
                image=image,
                label=output_text,
                width=width,
                height=height,
                name=output_path,
            )

            file_count += 1

    logger.info(f"Finished. Saved {file_count} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
