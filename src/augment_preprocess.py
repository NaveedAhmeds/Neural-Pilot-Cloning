# src/augment_preprocess.py

import cv2
import numpy as np
import random


# ---------- Preprocessing ----------

def crop_image(img, top=60, bottom=25):
    """
    Crop the image to remove sky and car hood.
    img: H x W x 3 (BGR as read by cv2).
    """
    h, w = img.shape[:2]
    return img[top:h-bottom, 0:w]


def resize_image(img, new_size=(200, 66)):
    """
    Resize image to NVIDIA input size (width=200, height=66).
    """
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def bgr_to_yuv(img):
    """
    Convert BGR image (cv2 default) to YUV color space.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def normalize_image(img):
    """
    Normalize image to [0, 1] float32.
    """
    return img.astype(np.float32) / 255.0


def gaussian_blur(img, ksize=(3, 3)):
    """
    Optional Gaussian blur to reduce noise.
    """
    return cv2.GaussianBlur(img, ksize, 0)


def preprocess(img):
    """
    Full preprocessing pipeline:
    - crop road area
    - resize to 200x66
    - convert to YUV
    - optional blur
    - normalize
    Returns float32 image of shape (66, 200, 3).
    """
    img = crop_image(img)
    img = resize_image(img, (200, 66))
    img = bgr_to_yuv(img)
    img = gaussian_blur(img)
    img = normalize_image(img)
    return img


# ---------- Augmentation (training only) ----------

def random_flip(img, steering):
    """
    Random horizontal flip with steering sign inversion.
    """
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def random_brightness(img):
    """
    Random brightness adjustment in HSV space.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ratio = 0.5 + random.random()  # [0.5, 1.5)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def random_shift(img, steering, range_x=50, range_y=10):
    """
    Random horizontal/vertical shift with slight steering adjustment.
    Positive x shift -> steering slightly more right.
    """
    h, w = img.shape[:2]
    tx = range_x * (random.random() - 0.5)
    ty = range_y * (random.random() - 0.5)

    steering += tx * 0.002  # tweak factor

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h))
    return img, steering


def random_shadow(img):
    """
    Add a random shadow region.
    """
    h, w = img.shape[:2]
    x1, y1 = w * np.random.rand(), 0
    x2, y2 = w * np.random.rand(), h
    xm, ym = np.mgrid[0:h, 0:w]

    mask = np.zeros_like(img[:, :, 1])
    mask[((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1)) > 0] = 1

    rand = 0.5 + 0.5 * np.random.rand()
    cond = mask == np.random.randint(2)
    img[:, :, 0][cond] = img[:, :, 0][cond] * rand
    img[:, :, 1][cond] = img[:, :, 1][cond] * rand
    img[:, :, 2][cond] = img[:, :, 2][cond] * rand

    return img


def augment(img, steering):
    """
    Apply a set of random augmentations for training images only.
    """
    # All augmentations operate on BGR images before preprocessing
    img, steering = random_shift(img, steering)
    img, steering = random_flip(img, steering)
    if random.random() < 0.5:
        img = random_brightness(img)
    if random.random() < 0.5:
        img = random_shadow(img)
    return img, steering


# -------    Testing    --------#


if __name__ == "__main__":
    import os
    import cv2

    # pick one sample image from your dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_img_path = os.path.join(project_root, "data", "IMG")

    # just grab any jpg from IMG
    files = [f for f in os.listdir(sample_img_path) if f.lower().endswith(".jpg")]
    if not files:
        raise FileNotFoundError("No .jpg files found in data/IMG")
    img_path = os.path.join(sample_img_path, files[0])

    img = cv2.imread(img_path)
    steering = 0.0

    # test augment + preprocess
    img_aug, steering_aug = augment(img.copy(), steering)
    img_proc = preprocess(img_aug)

    print("Original shape:", img.shape)
    print("After preprocess shape:", img_proc.shape)
    print("Augmented steering:", steering_aug)
