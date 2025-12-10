# src/data_loader.py

import os
import csv

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_samples(csv_path, img_dir):
    """
    Load (image_path, steering) samples from Udacity-style driving_log.csv.
    """
    samples = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for line in reader:
            # crude header check
            if not (line[0].lower().endswith(".jpg") or line[0].lower().endswith(".png")):
                continue

            center_path = line[0].strip()
            steering = float(line[3])

            # Use only filename from CSV and join with img_dir
            filename = os.path.basename(center_path)
            full_img_path = os.path.join(img_dir, filename)
            samples.append((full_img_path, steering))
    return samples


def train_val_split(samples, test_size=0.2, random_state=42):
    train_samples, val_samples = train_test_split(
        samples,
        test_size=test_size,
        shuffle=True,
        random_state=random_state
    )
    return train_samples, val_samples


def plot_steering_histogram(samples, bins=31, save_path=None):
    steerings = [s[1] for s in samples]
    plt.figure(figsize=(6, 4))
    plt.hist(steerings, bins=bins)
    plt.xlabel("Steering angle")
    plt.ylabel("Count")
    plt.title("Steering angle distribution")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # project_root/
    #   data/
    #     driving_log.csv
    #     IMG/
    #   src/
    #     data_loader.py

    this_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(this_file)
    project_root = os.path.dirname(src_dir)

    data_dir = os.path.join(project_root, "data")
    csv_path = os.path.join(data_dir, "driving_log.csv")
    img_dir = os.path.join(data_dir, "IMG")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find driving_log.csv at {csv_path}")
    if not os.path.isdir(img_dir):
        raise NotADirectoryError(f"Could not find IMG directory at {img_dir}")

    samples = load_samples(csv_path, img_dir)
    train_samples, val_samples = train_val_split(samples)

    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    docs_dir = os.path.join(project_root, "docs")
    hist_path = os.path.join(docs_dir, "steering_hist.png")
    plot_steering_histogram(samples, bins=31, save_path=hist_path)
    print(f"Steering histogram saved to: {hist_path}")
