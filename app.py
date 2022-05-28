#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import sys
import tarfile

sys.path.insert(0, 'yolov5_anime')

import cv2
import gradio as gr
import huggingface_hub
import numpy as np
import PIL.Image
import torch
from models.yolo import Model
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

ORIGINAL_REPO_URL = 'https://github.com/zymk9/yolov5_anime'
TITLE = 'zymk9/yolov5_anime'
DESCRIPTION = f'A demo for {ORIGINAL_REPO_URL}'
ARTICLE = None

TOKEN = os.environ['TOKEN']
MODEL_REPO = 'hysts/yolov5_anime'
MODEL_FILENAME = 'yolov5x_anime.pth'
CONFIG_FILENAME = 'yolov5x.yaml'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--score-slider-step', type=float, default=0.05)
    parser.add_argument('--score-threshold', type=float, default=0.4)
    parser.add_argument('--iou-slider-step', type=float, default=0.05)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        dataset_repo = 'hysts/sample-images-TADNE'
        path = huggingface_hub.hf_hub_download(dataset_repo,
                                               'images.tar.gz',
                                               repo_type='dataset',
                                               use_auth_token=TOKEN)
        with tarfile.open(path) as f:
            f.extractall()
    return sorted(image_dir.glob('*'))


def load_model(device: torch.device) -> torch.nn.Module:
    torch.set_grad_enabled(False)
    model_path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                                 MODEL_FILENAME,
                                                 use_auth_token=TOKEN)
    config_path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                                  CONFIG_FILENAME,
                                                  use_auth_token=TOKEN)
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


def main():
    args = parse_args()
    device = torch.device(args.device)

    image_paths = load_sample_image_paths()
    examples = [[path.as_posix(), args.score_threshold, args.iou_threshold]
                for path in image_paths]

    model = load_model(device)

    func = functools.partial(predict, device=device, model=model)
    func = functools.update_wrapper(func, predict)

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='pil', label='Input'),
            gr.inputs.Slider(0,
                             1,
                             step=args.score_slider_step,
                             default=args.score_threshold,
                             label='Score Threshold'),
            gr.inputs.Slider(0,
                             1,
                             step=args.iou_slider_step,
                             default=args.iou_threshold,
                             label='IoU Threshold'),
        ],
        gr.outputs.Image(label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
