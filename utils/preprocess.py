def filter_invalid_bboxes(dataset):
    for idx in range(len(dataset)):
        sample = dataset[idx]
        valid_bboxes = []
        valid_labels = []
        bboxes = sample[1]["boxes"]
        labels = sample[1]["class_labels"]
        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox[:4]
            if x_min < x_max and y_min < y_max:
                valid_bboxes.append(bbox)
                valid_labels.append(labels[i])
            else:
                print(
                    f"Invalid bbox detected in {idx}. sample (Label: {labels[i]}) and discarded: {bbox}"
                )

        dataset[idx][1]["boxes"] = valid_bboxes
        dataset[idx][1]["class_labels"] = valid_labels

    return dataset
