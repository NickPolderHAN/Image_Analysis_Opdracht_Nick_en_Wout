import keras
from keras import layers, Sequential, Input
from medmnist import ChestMNIST
import tensorflow as tf
import numpy as np

def get_dataset():
    # The pictures and labels for the train set.
    pictures_train = []
    labels_train = []

    # The pictures and labels for the test set.
    pictures_test = []
    labels_test = []

    # Data ophalen en splitsen in train en test sets.
    dataset_train = ChestMNIST(split='train', download=True, size=128)
    dataset_test = ChestMNIST(split='test', download=True, size=128)

    # Amount of labels.
    label_count = 14

    # The training set.
    for i in dataset_train:
        if 1 in i[1]:
            pictures_train.append(np.array(i[0]))  # Convert PIL image to numpy array
            labels_train.append(1)
        else:
            pictures_train.append(np.array(i[0]))  # Convert PIL image to numpy array
            labels_train.append(0)

    # The test set.
    for i in dataset_test:
        if 1 in i[1]:
            pictures_test.append(np.array(i[0]))  # Convert PIL image to numpy array
            labels_test.append(1)
        else:
            pictures_test.append(np.array(i[0]))  # Convert PIL image to numpy array
            labels_test.append(0)

    return pictures_train, labels_train, pictures_test, labels_test


def merge_picture_and_label(picture_train, labels_train, picture_test, labels_test):
    batch_size = 100
    train_ds_ = tf.data.Dataset.from_tensor_slices((np.array(picture_train), np.array(labels_train)))  # Convert to numpy array
    test_ds_ = tf.data.Dataset.from_tensor_slices((np.array(picture_test), np.array(labels_test)))  # Convert to numpy array
    train_ds_ = train_ds_.batch(batch_size)
    test_ds_ = test_ds_.batch(batch_size)
    return train_ds_, test_ds_


def model_training(train_data, test_data):
    img_height = 128
    img_width = 128

    print(train_data)

    data_augmentation = keras.Sequential(
        [
            # layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        Input(shape=(img_height, img_width, 1)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(4, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    # Fitting the model.
    epochs = 20
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs
    )


if __name__ == '__main__':
    p_train, label_train, p_test, label_test = get_dataset()
    train_data, test_data = merge_picture_and_label(p_train, label_train, p_test, label_test)
    model_training(train_data, test_data)
