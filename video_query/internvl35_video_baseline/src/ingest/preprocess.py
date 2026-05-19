"""
InternVL 공식 build_transform / dynamic_preprocess를 그대로 사용.
video용으로 max_num=1 강제 (타일 1개).
"""
from __future__ import annotations
from typing import List
import math
from PIL import Image
import torchvision.transforms as T
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_video_transform(image_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def dynamic_preprocess_video(
    image: Image.Image,
    image_size: int = 448,
    max_num: int = 1,
) -> List[Image.Image]:
    """video 모드: max_num=1이므로 타일 분할 없이 단순 resize만 수행."""
    return [image.resize((image_size, image_size), Image.BICUBIC)]
