# src/model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, ELU

def nvidia_model(input_shape=(66, 200, 3)):
    """
    NVIDIA/PilotNet-style CNN for steering angle regression.
    """
    model = Sequential()

    # Normalization (images already in [0,1], but keep for safety)
    model.add(Lambda(lambda x: x, input_shape=input_shape))

    # Conv layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    # Fully connected
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))  # steering angle

    model.compile(loss='mse', optimizer='adam')
    return model
