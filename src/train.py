# src/train.py

import os
from math import ceil

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

from data_loader import load_samples, train_val_split
from generator import batch_generator
from model import nvidia_model


def main():
    # Paths
    this_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(this_file)
    project_root = os.path.dirname(src_dir)

    data_dir = os.path.join(project_root, "data")
    csv_path = os.path.join(data_dir, "driving_log.csv")
    img_dir = os.path.join(data_dir, "IMG")

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load and split data
    samples = load_samples(csv_path, img_dir)
    train_samples, val_samples = train_val_split(samples)

    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    # Generators
    batch_size = 32
    train_gen = batch_generator(train_samples, batch_size=batch_size, training=True)
    val_gen = batch_generator(val_samples, batch_size=batch_size, training=False)

    # Model
    model = nvidia_model(input_shape=(66, 200, 3))
    model.summary()

    # Callbacks
    checkpoint_path = os.path.join(models_dir, "model_best.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=1,
        restore_best_weights=True,
    )

    # Training configuration
    epochs = 8
    steps_per_epoch = ceil(len(train_samples) / batch_size)
    val_steps = ceil(len(val_samples) / batch_size)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )

    # Save final model
    final_path = os.path.join(models_dir, "model.h5")
    model.save(final_path)
    print(f"Final model saved to: {final_path}")
    print(f"Best model (by val_loss) at: {checkpoint_path}")


if __name__ == "__main__":
    main()
