# src/generator.py

import os
import cv2
import numpy as np
from sklearn.utils import shuffle

from data_loader import load_samples, train_val_split
from augment_preprocess import augment, preprocess


def batch_generator(samples, batch_size=32, training=True):
    """
    Generator yielding (X, y) batches.
    - training=True: apply augmentation + preprocessing.
    - training=False: only preprocessing (no augmentation).
    """
    num_samples = len(samples)
    while True:  # Keras generators loop forever
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []

            for img_path, steering in batch_samples:
                # read image (BGR)
                img = cv2.imread(img_path)
                if img is None:
                    # skip if file missing / unreadable
                    continue

                steer = steering

                if training:
                    img, steer = augment(img, steer)

                img = preprocess(img)  # (66, 200, 3), float32

                images.append(img)
                angles.append(steer)

            if len(images) == 0:
                continue

            X = np.array(images, dtype=np.float32)
            y = np.array(angles, dtype=np.float32)

            yield X, y


if __name__ == "__main__":
    """
    Quick smoke test to ensure generator runs.
    """
    this_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(this_file)
    project_root = os.path.dirname(src_dir)

    data_dir = os.path.join(project_root, "data")
    csv_path = os.path.join(data_dir, "driving_log.csv")
    img_dir = os.path.join(data_dir, "IMG")

    samples = load_samples(csv_path, img_dir)
    train_samples, val_samples = train_val_split(samples)

    gen = batch_generator(train_samples, batch_size=16, training=True)

    X_batch, y_batch = next(gen)
    print("Batch X shape:", X_batch.shape)  # expect (<=16, 66, 200, 3)
    print("Batch y shape:", y_batch.shape)
    print("Example steering angles:", y_batch[:5])
