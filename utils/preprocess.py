import albumentations as A
import numpy as np


def filter_invalid_bboxes(example):
    valid_bboxes = []
    valid_bbox_ids = []
    valid_categories = []
    valid_areas = []

    for i, bbox in enumerate(example["objects"]["bbox"]):
        x_min, y_min, x_max, y_max = bbox[:4]
        if x_min < x_max and y_min < y_max:
            valid_bboxes.append(bbox)
            valid_bbox_ids.append(example["objects"]["bbox_id"][i])
            valid_categories.append(example["objects"]["category"][i])
            valid_areas.append(example["objects"]["area"][i])
        else:
            print(
                f"Image with invalid bbox: {example['image_id']} Invalid bbox detected and discarded: {bbox} - bbox_id: {example['objects']['bbox_id'][i]} - category: {example['objects']['category'][i]}"
            )

    example["objects"]["bbox"] = valid_bboxes
    example["objects"]["bbox_id"] = valid_bbox_ids
    example["objects"]["category"] = valid_categories
    example["objects"]["area"] = valid_areas


def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


def convert_voc_to_coco(bbox):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    return [xmin, ymin, width, height]


def transform_aug_ann(examples, transform, image_processor):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])

        # Convert to COCO format
        converted_bboxes = [convert_voc_to_coco(bbox) for bbox in out["bboxes"]]
        bboxes.append(converted_bboxes)

        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


def transform_train(examples):
    train_transform = A.Compose(
        [
            A.LongestMaxSize(500),
            A.PadIfNeeded(500, 500, border_mode=0, value=(0, 0, 0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.GaussianBlur(p=0.5),
            A.GaussNoise(p=0.5),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category"]),
    )

    return transform_aug_ann(examples, transform=train_transform)


def transform_val(examples):
    val_transform = A.Compose(
        [
            A.LongestMaxSize(500),
            A.PadIfNeeded(500, 500, border_mode=0, value=(0, 0, 0)),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category"]),
    )

    return transform_aug_ann(examples, transform=val_transform)
