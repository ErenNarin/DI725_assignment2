import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from utils.visualize import visualize_predictions


def evaluate_model(model, dataset, processor, id2label, batch_size=1, max_samples=None, show_samples=5):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    metric = MeanAveragePrecision(class_metrics=True)
    sample_count = 0

    with torch.no_grad():
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
    print(f"\n mAP = {results['map']:.4f}")

    print("\n Per-Class Average Precision (AP):")
    for class_idx, ap in enumerate(results["map_per_class"]):
        label = id2label.get(class_idx, f"class_{class_idx}")
        print(f"  {label:15s} : {ap:.4f}")

    return results
