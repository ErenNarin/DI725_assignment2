import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_labels(image, target, id2label=None):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(image)

    for box, label in zip(target["boxes"], target["class_labels"]):
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


def label_histogram(dataset, id2label):
    label_examples = {}
    for example in dataset:
        for label in example[1]["class_labels"]:
            label = int(label)
            if id2label[label] not in label_examples:
                label_examples[id2label[label]] = 1
            else:
                label_examples[id2label[label]] += 1

    categories = list(label_examples.keys())
    values = list(label_examples.values())

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.bar(categories, values, color="skyblue")

    ax.set_xlabel("Categories", fontsize=14)
    ax.set_ylabel("Number of Occurrences", fontsize=14)
    ax.set_title("Number of Occurrences by Category", fontsize=16)

    ax.set_xticklabels(categories, rotation=90, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()
