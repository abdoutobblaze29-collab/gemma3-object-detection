# convert_plantdoc_yolov8_to_paligemma.py

import argparse
from pathlib import Path

import yaml
from PIL import Image
from datasets import Dataset, DatasetDict, Image as HFImage


def yolo_to_coco_bbox(yolo_bbox, image_width, image_height):
    """
    YOLOv8 bbox:
        x_center, y_center, width, height
        all normalized between 0 and 1

    COCO bbox:
        x_min, y_min, width, height
        in pixels
    """
    x_center, y_center, box_width, box_height = yolo_bbox

    x_center *= image_width
    y_center *= image_height
    box_width *= image_width
    box_height *= image_height

    x_min = x_center - box_width / 2
    y_min = y_center - box_height / 2

    return [x_min, y_min, box_width, box_height]


def coco_to_xyxy(coco_bbox):
    x, y, width, height = coco_bbox
    return [x, y, x + width, y + height]


def format_location(value, max_value):
    value = max(0, min(value, max_value))
    loc = int(round(value * 1024 / max_value))
    loc = max(0, min(loc, 1024))
    return f"<loc{loc:04}>"


def bbox_to_detection_string(coco_bbox, class_name, image_width, image_height):
    x1, y1, x2, y2 = coco_to_xyxy(coco_bbox)

    return (
        format_location(y1, image_height)
        + format_location(x1, image_width)
        + format_location(y2, image_height)
        + format_location(x2, image_width)
        + f" {class_name}"
    )


def read_data_yaml(dataset_dir):
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml at {data_yaml}")

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]

    return names


def find_images(images_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def convert_split(dataset_dir, split_name, class_names):
    images_dir = dataset_dir / split_name / "images"
    labels_dir = dataset_dir / split_name / "labels"

    if not images_dir.exists():
        return None

    rows = []

    for image_path in find_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as img:
            image_width, image_height = img.size

        detection_strings = []
        objects = {
            "bbox": [],
            "category": [],
            "category_name": [],
        }

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(f"Invalid YOLO label in {label_path}: {line}")

                class_id = int(parts[0])
                yolo_bbox = list(map(float, parts[1:]))

                class_name = class_names[class_id]
                coco_bbox = yolo_to_coco_bbox(
                    yolo_bbox,
                    image_width,
                    image_height,
                )

                detection_strings.append(
                    bbox_to_detection_string(
                        coco_bbox,
                        class_name,
                        image_width,
                        image_height,
                    )
                )

                objects["bbox"].append(coco_bbox)
                objects["category"].append(class_id)
                objects["category_name"].append(class_name)

        rows.append(
            {
                "image": str(image_path),
                "width": image_width,
                "height": image_height,
                "objects": objects,
                "label_for_paligemma": " ; ".join(detection_strings),
            }
        )

    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("image", HFImage())

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to Roboflow YOLOv8 export folder containing data.yaml",
    )
    parser.add_argument(
        "--output_repo",
        type=str,
        default=None,
        help="Optional Hugging Face Hub repo ID to push to",
    )
    parser.add_argument(
        "--save_to_disk",
        type=str,
        default=None,
        help="Optional local path to save the Hugging Face DatasetDict",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    class_names = read_data_yaml(dataset_dir)

    split_map = {
        "train": "train",
        "valid": "validation",
        "val": "validation",
        "test": "test",
    }

    dataset_dict = {}

    for roboflow_split, hf_split in split_map.items():
        split_dataset = convert_split(dataset_dir, roboflow_split, class_names)

        if split_dataset is not None:
            dataset_dict[hf_split] = split_dataset
            print(f"[INFO] Converted {roboflow_split} -> {hf_split}: {len(split_dataset)} images")

    dataset = DatasetDict(dataset_dict)

    if args.save_to_disk:
        dataset.save_to_disk(args.save_to_disk)
        print(f"[INFO] Saved dataset to {args.save_to_disk}")

    if args.output_repo:
        dataset.push_to_hub(args.output_repo)
        print(f"[INFO] Pushed dataset to {args.output_repo}")


if __name__ == "__main__":
    main()
