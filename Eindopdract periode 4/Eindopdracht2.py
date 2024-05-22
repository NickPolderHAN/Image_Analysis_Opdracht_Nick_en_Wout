import keras
from keras import layers, Sequential, Input
from medmnist import ChestMNIST
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def get_dataset():
    # Lists to store pictures and labels
    pictures = []
    labels = []

    # Data ophalen en splitsen in train en test sets.
    dataset_train = ChestMNIST(split='train', download=True, size=128)
    dataset_test = ChestMNIST(split='test', download=True, size=128)

    # Populate pictures and labels for training
    for image, label in dataset_train:
        pictures.append(np.array(image))
        labels.append(1 if 1 in label else 0)

    # Split the data into training, validation, and test sets
    pictures_train, pictures_remaining, labels_train, labels_remaining = train_test_split(pictures, labels, test_size=0.2, random_state=42)

    # Split the remaining data into validation and test sets
    pictures_valid, pictures_test, labels_valid, labels_test = train_test_split(pictures_remaining, labels_remaining, test_size=0.5, random_state=42)

    return pictures_train, labels_train, pictures_valid, labels_valid, pictures_test, labels_test


def merge_picture_and_label(pictures_train, labels_train, pictures_valid, labels_valid, pictures_test, labels_test):
    batch_size = 100
    # Convert to dataset.
    train_ds_ = tf.data.Dataset.from_tensor_slices((np.array(pictures_train), np.array(labels_train)))  # Convert to numpy array
    test_ds_ = tf.data.Dataset.from_tensor_slices((np.array(pictures_test), np.array(labels_test)))  # Convert to numpy array
    val_ds_ = tf.data.Dataset.from_tensor_slices((np.array(pictures_valid), np.array(labels_valid)))

    # Set batch size.
    train_ds_ = train_ds_.batch(batch_size)
    test_ds_ = test_ds_.batch(batch_size)
    val_ds_ = val_ds_.batch(batch_size)

    return train_ds_, test_ds_, val_ds_


def model_training(train_data, test_data):
    img_height = 128
    img_width = 128

    # Creating data augmentation layers
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    # Building the model
    model = Sequential([
        data_augmentation,
        Input(shape=(img_height, img_width, 1)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(4, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation="sigmoid")
    ])

    # Compiling the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Now call build to ensure the model is built
    model.build((None, img_height, img_width, 1))

    # Print model summary
    model.summary()

    # Fitting the model.
    epochs = 5
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs
    )


if __name__ == '__main__':
    pictures_train, labels_train, pictures_valid, labels_valid, pictures_test, labels_test = get_dataset()
    train_data, test_data, val_ds = merge_picture_and_label(pictures_train,
                                                            labels_train,
                                                            pictures_valid,
                                                            labels_valid,
                                                            pictures_test,
                                                            labels_test)
    model_training(train_data, test_data)
