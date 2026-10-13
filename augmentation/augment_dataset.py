from pathlib import Path
import shutil
import cv2
import albumentations as A


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

INPUT_IMAGES = DATASET_DIR / "images" / "train"
INPUT_LABELS = DATASET_DIR / "labels" / "train"

OUTPUT_IMAGES = DATASET_DIR / "images_augmented" / "train"
OUTPUT_LABELS = DATASET_DIR / "labels_augmented" / "train"

COPIES_PER_IMAGE = 2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# Augmentation pipeline
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.7
        ),
        A.Rotate(
            limit=5,
            p=0.4,
            border_mode=cv2.BORDER_CONSTANT,
            fill=114
        ),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3
        )
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_ids"],
        min_visibility=0.2
    )
)


def read_yolo_labels(label_path):
    boxes = []
    class_ids = []

    with label_path.open("r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])

            boxes.append([x_center, y_center, width, height])
            class_ids.append(class_id)

    return boxes, class_ids


def write_yolo_labels(label_path, boxes, class_ids):
    with label_path.open("w") as file:
        for class_id, box in zip(class_ids, boxes):
            x_center, y_center, width, height = box

            file.write(
                f"{class_id} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}\n"
            )


def main():
    if not INPUT_IMAGES.exists() or not INPUT_LABELS.exists():
        raise FileNotFoundError(
            "Input image or label folder not found. "
            f"Check:\n{INPUT_IMAGES}\n{INPUT_LABELS}"
        )

    # Avoid accidentally mixing outputs from previous runs.
    if OUTPUT_IMAGES.exists() and any(OUTPUT_IMAGES.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_IMAGES} is not empty. "
            "Move or remove the existing output before running again."
        )

    if OUTPUT_LABELS.exists() and any(OUTPUT_LABELS.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_LABELS} is not empty. "
            "Move or remove the existing output before running again."
        )

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in INPUT_IMAGES.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise ValueError(f"No images found in {INPUT_IMAGES}")

    original_count = 0
    augmented_count = 0
    skipped_count = 0

    for image_path in images:
        label_path = INPUT_LABELS / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"Skipping image without label: {image_path.name}")
            skipped_count += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            skipped_count += 1
            continue

        boxes, class_ids = read_yolo_labels(label_path)

        # Copy the original image and its labels.
        original_image_path = OUTPUT_IMAGES / image_path.name
        original_label_path = OUTPUT_LABELS / label_path.name

        shutil.copy2(image_path, original_image_path)
        shutil.copy2(label_path, original_label_path)
        original_count += 1

        # Convert OpenCV's BGR image to RGB for Albumentations.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for copy_index in range(1, COPIES_PER_IMAGE + 1):
            try:
                result = transform(
                    image=image_rgb,
                    bboxes=boxes,
                    class_ids=class_ids
                )

                augmented_image = cv2.cvtColor(
                    result["image"],
                    cv2.COLOR_RGB2BGR
                )

                augmented_boxes = result["bboxes"]
                augmented_class_ids = result["class_ids"]

                output_stem = (
                    f"{image_path.stem}_aug{copy_index}"
                )

                output_image_path = (
                    OUTPUT_IMAGES / f"{output_stem}{image_path.suffix}"
                )
                output_label_path = (
                    OUTPUT_LABELS / f"{output_stem}.txt"
                )

                if not cv2.imwrite(
                    str(output_image_path),
                    augmented_image
                ):
                    raise OSError(
                        f"Could not save {output_image_path}"
                    )

                write_yolo_labels(
                    output_label_path,
                    augmented_boxes,
                    augmented_class_ids
                )

                augmented_count += 1

            except Exception as error:
                print(
                    f"Failed to augment {image_path.name}, "
                    f"copy {copy_index}: {error}"
                )
                skipped_count += 1

        print(f"Processed: {image_path.name}")

    print("\nAugmentation complete!")
    print(f"Original images copied: {original_count}")
    print(f"Augmented copies created: {augmented_count}")
    print(f"Skipped items or failed copies: {skipped_count}")
    print(f"Total output images: {original_count + augmented_count}")
    print(f"Output images: {OUTPUT_IMAGES}")
    print(f"Output labels: {OUTPUT_LABELS}")


if __name__ == "__main__":
    main()