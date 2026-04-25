import os
from functools import partial

import torch
import albumentations as A
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from config import Configuration
from utils import test_collate_function, visualize_bounding_boxes

os.makedirs("outputs", exist_ok=True)


def get_augmentations(cfg):
    resize_size = 512 if "SmolVLM" in cfg.model_id else 896

    return A.Compose([
        A.Resize(height=resize_size, width=resize_size),
    ])


def get_dataloader(processor, cfg):
    test_dataset = load_dataset(cfg.dataset_id, split="test")

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


if __name__ == "__main__":
    cfg = Configuration()

    processor = AutoProcessor.from_pretrained(cfg.checkpoint_id)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.checkpoint_id,
        torch_dtype=cfg.dtype,
        device_map="auto",
    )

    model.eval()

    test_dataloader = get_dataloader(processor=processor, cfg=cfg)

    file_count = 0

    for sample, sample_images in test_dataloader:
        predictions = predict_batch(
            model=model,
            processor=processor,
            sample=sample,
            max_new_tokens=128,
        )

        for output_text, sample_image in zip(predictions, sample_images):
            image = sample_image[0]

            print("Prediction:", output_text)

            width, height = image.size

            visualize_bounding_boxes(
                image=image,
                label=output_text,
                width=width,
                height=height,
                name=f"outputs/output_{file_count}.png",
            )

            file_count += 1
