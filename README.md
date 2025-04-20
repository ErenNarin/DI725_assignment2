# DI725_assignment2

## Abstract

In this work, the aim is to compare the performance of transformer-based object detection models with that
of YOLOV3-Tiny and MobileNetV2-SSDLite models. To do this,
traffic images, which are taken from unmanned aerial vehicles,
are used. As evaluation metrics, average precision values by
class and the mean average precision values are used. As a
transformer-based object detection model, DETR was selected
and fine-tuned with traffic images. According to the results,
the fine-tuned model underperformed the reference models’
performances. The model predicted classes with high accuracy,
but failed on the localization task.

## URLs

Training Notebook: [fine-tune_detr_training.ipynb](fine-tune_detr_training.ipynb)

Evaluation Notebook: [fine-tune_detr_evaluation.ipynb](fine-tune_detr_evaluation.ipynb)

W&B Project: [DI725_assignment_2_2389088_detr-auair](https://wandb.ai/erennarin-92-metu-middle-east-technical-university/DI725_assignment_2_2389088_detr-auair)

## Execution Instructions

> **_NOTE:_** Before executing notebook files, the following installation commands should be executed to prepare work environment.
> 
> ```console
> pip install -U -q datasets transformers[torch] timm wandb torchmetrics matplotlib albumentations numpy scikit-learn
> ```

> **_NOTE:_** Before executing the training notebook, dataset should be moved under "data/" path. The correct organization should be like this:
> 
> data/auair2019/images/frame_*.jpg -> Input images
> 
> data/auair2019/annotations.json -> Annotations in json format

To execute end-to-end process, notebooks should be run in this order:

1. Execute [fine-tune_detr_training.ipynb](fine-tune_detr_training.ipynb). This notebook will load and preprocess data, fine-tune the DETR model, and save the fine-tuned model alongside the processor in the local storage.
2. Execute [fine-tune_detr_evaluation.ipynb](fine-tune_detr_evaluation.ipynb). This notebook will load the test data, load the fine-tuned model, and evaluate the model. Metrics will be printed at the end of the notebook. Also, results are discussed in this notebook.