#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import pathlib
import sys
import tarfile

import cv2
import gradio as gr
import huggingface_hub
import numpy as np
import PIL.Image
import torch

sys.path.insert(0, 'yolov5_anime')

from models.yolo import Model
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

TITLE = 'zymk9/yolov5_anime'
DESCRIPTION = 'This is an unofficial demo for https://github.com/zymk9/yolov5_anime.'

HF_TOKEN = os.getenv('HF_TOKEN')
MODEL_REPO = 'hysts/yolov5_anime'
MODEL_FILENAME = 'yolov5x_anime.pth'
CONFIG_FILENAME = 'yolov5x.yaml'


def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        dataset_repo = 'hysts/sample-images-TADNE'
        path = huggingface_hub.hf_hub_download(dataset_repo,
                                               'images.tar.gz',
                                               repo_type='dataset',
                                               use_auth_token=HF_TOKEN)
        with tarfile.open(path) as f:
            f.extractall()
    return sorted(image_dir.glob('*'))


def load_model(device: torch.device) -> torch.nn.Module:
    torch.set_grad_enabled(False)
    model_path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                                 MODEL_FILENAME,
                                                 use_auth_token=HF_TOKEN)
    config_path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                                  CONFIG_FILENAME,
                                                  use_auth_token=HF_TOKEN)
    state_dict = torch.load(model_path)
    model = Model(cfg=config_path)
    model.load_state_dict(state_dict)
    model.to(device)
    if device.type != 'cpu':
        model.half()
    model.eval()
    return model


@torch.inference_mode()
def predict(image: PIL.Image.Image, score_threshold: float,
            iou_threshold: float, device: torch.device,
            model: torch.nn.Module) -> np.ndarray:
    orig_image = np.asarray(image)

    image = letterbox(orig_image, new_shape=640)[0]
    data = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255
    data = data.to(device).unsqueeze(0)
    if device.type != 'cpu':
        data = data.half()

    preds = model(data)[0]
    preds = non_max_suppression(preds, score_threshold, iou_threshold)

    detections = []
    for pred in preds:
        if pred is not None and len(pred) > 0:
            pred[:, :4] = scale_coords(data.shape[2:], pred[:, :4],
                                       orig_image.shape).round()
            # (x0, y0, x1, y0, conf, class)
            detections.append(pred.cpu().numpy())
    detections = np.concatenate(detections) if detections else np.empty(
        shape=(0, 6))

    res = orig_image.copy()
    for det in detections:
        x0, y0, x1, y1 = det[:4].astype(int)
        cv2.rectangle(res, (x0, y0), (x1, y1), (0, 255, 0), 3)
    return res


image_paths = load_sample_image_paths()
examples = [[path.as_posix(), 0.4, 0.5] for path in image_paths]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model(device)
func = functools.partial(predict, device=device, model=model)

gr.Interface(
    fn=func,
    inputs=[
        gr.Image(label='Input', type='pil'),
        gr.Slider(label='Score Threshold',
                  minimum=0,
                  maximum=1,
                  step=0.05,
                  value=0.4),
        gr.Slider(label='IoU Threshold',
                  minimum=0,
                  maximum=1,
                  step=0.05,
                  value=0.5),
    ],
    outputs=gr.Image(label='Output'),
    examples=examples,
    title=TITLE,
    description=DESCRIPTION,
).queue().launch(show_api=False)
