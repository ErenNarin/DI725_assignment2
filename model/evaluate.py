import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_predictions(image, preds, target, id2label=None):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(image)

    for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
        if score > 0.8:
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


def evaluate_model(model, dataset, processor, id2label, batch_size=1, max_samples=None, show_samples=5):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    metric = MeanAveragePrecision()
    sample_count = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        for sample in batch:
            image, target = sample
            image = image.convert("RGB")
            encoding = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**encoding)

            postprocessed = processor.post_process_object_detection(
                outputs, target_sizes=[(image.height, image.width)], threshold=0.0
            )[0]

            preds = [{
                "boxes": postprocessed["boxes"].cpu(),
                "scores": postprocessed["scores"].cpu(),
                "labels": postprocessed["labels"].cpu()
            }]

            target_boxes = target["boxes"] * torch.tensor(
                [image.width, image.height, image.width, image.height]
            )
            target_dict = {
                "boxes": target_boxes,
                "labels": target["class_labels"]
            }

            metric.update(preds, [target_dict])

            if sample_count < show_samples:
                visualize_predictions(image, preds[0], target_dict, id2label=id2label)
                sample_count += 1

            if max_samples and sample_count >= max_samples:
                break

        if max_samples and sample_count >= max_samples:
            break

    results = metric.compute()
    print(f"\n [mAP@IoU=0.50:0.95] = {results['map']:.4f}")

    print("\n Per-Class Average Precision (AP):")
    for class_idx, ap in enumerate(results["map_per_class"]):
        label = id2label.get(class_idx, f"class_{class_idx}")
        print(f"  {label:15s} : {ap:.4f}")

    return results
