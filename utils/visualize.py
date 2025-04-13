import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw


def visualize_labels(image, target, id2label=None):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(image)

    for box, label in zip(target["boxes"], target["labels"]):
        x0, y0, x1, y1 = box.tolist()
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 linewidth=2, edgecolor="red", facecolor="none", linestyle="--")
        ax.add_patch(rect)
        name = id2label[label.item()] if id2label else str(label.item())
        ax.text(x0, y1, f"GT: {name}", color="red", fontsize=10)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def visualize_predictions(image, preds, target, id2label=None, threshold=0.8):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(image)

    for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
        if score > threshold:
            x0, y0, x1, y1 = box.tolist()
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            name = id2label[label.item()] if id2label else str(label.item())
            ax.text(x0, y0 - 4, f"{name}: {score:.2f}", color="lime", fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

    for box, label in zip(target["boxes"], target["labels"]):
        x0, y0, x1, y1 = box.tolist()
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 linewidth=2, edgecolor="red", facecolor="none", linestyle="--")
        ax.add_patch(rect)
        name = id2label[label.item()] if id2label else str(label.item())
        ax.text(x0, y1 + 4, f"GT: {name}", color="red", fontsize=10)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def draw_image_from_idx(dataset, idx, id2label):
    sample = dataset[idx]
    image = sample[0]
    annotations = sample["objects"]
    draw = ImageDraw.Draw(image)
    width, height = sample["width"], sample["height"]

    print(annotations)

    for i in range(len(annotations["bbox_id"])):
        box = annotations["bbox"][i]
        x1, y1, x2, y2 = tuple(box)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, y1), id2label[annotations["category"][i]], fill="green")

    return image


def plot_augmented_images(dataset, indices, transform=None):
    """
    Plot images and their annotations with optional augmentation.
    """
    num_rows = len(indices) // 3
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    for i, idx in enumerate(indices):
        row = i // num_cols
        col = i % num_cols

        # Draw augmented image
        image = draw_augmented_image_from_idx(dataset, idx, transform=transform)

        # Display image on the corresponding subplot
        axes[row, col].imshow(image)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def draw_augmented_image_from_idx(dataset, idx, id2label, transform=None):
    sample = dataset[idx]
    image = sample["image"]
    annotations = sample["objects"]

    # Convert image to RGB and NumPy array
    image = np.array(image.convert("RGB"))[:, :, ::-1]

    if transform:
        augmented = transform(image=image, bboxes=annotations["bbox"], category=annotations["category"])
        image = augmented["image"]
        annotations["bbox"] = augmented["bboxes"]
        annotations["category"] = augmented["category"]

    image = Image.fromarray(image[:, :, ::-1])  # Convert back to PIL Image
    draw = ImageDraw.Draw(image)
    width, height = sample["width"], sample["height"]

    for i in range(len(annotations["bbox_id"])):
        box = annotations["bbox"][i]
        x1, y1, x2, y2 = tuple(box)

        # Normalize coordinates if necessary
        if max(box) <= 1.0:
            x1, y1 = int(x1 * width), int(y1 * height)
            x2, y2 = int(x2 * width), int(y2 * height)
        else:
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, y1), id2label[annotations["category"][i]], fill="green")

    return image


def plot_results(image, results, threshold=0.6):
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for result in results:
        score = result["score"]
        label = result["label"]
        box = list(result["box"].values())

        if score > threshold:
            x1, y1, x2, y2 = tuple(box)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1 + 5, y1 - 10), label, fill="white")
            draw.text((x1 + 5, y1 + 10), f"{score:.2f}", fill="green" if score > 0.7 else "red")

    return image
