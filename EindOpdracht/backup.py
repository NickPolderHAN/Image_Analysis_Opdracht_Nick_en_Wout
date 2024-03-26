import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from PIL import Image
import h5py
import PIL
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def store_normalized_data(train_data, test_data):
    batch_size = 150
    img_height = 96
    img_width = 96

    train_data = train_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(4, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation="sigmoid")
    ])


    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.summary()

    epochs = 1
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    predictions = model.predict(test_data)
    # Assuming labels are binary (0 or 1)
    y_score = predictions.flatten()
    y_true = test_data.map(lambda x, y: y).unbatch()  # Extract true labels
    y_true = [label.numpy() for label in y_true]

    plot_roc_curve(y_true, y_score)

    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()


def normalize_and_configure(train_ds, labels_training, test_ds, labels_testing):
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    # Assuming train_ds and test_ds are NumPy arrays
    train_ds = tf.data.Dataset.from_tensor_slices((train_ds, labels_training))
    test_ds = tf.data.Dataset.from_tensor_slices((test_ds, labels_testing))
    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    store_normalized_data(normalized_train_ds, normalized_test_ds)


def configure_dataset():
    # x zijn de plaatjes, y zijn de binaire labels (wel of niet)
    file_train_pictures = h5py.File("train/camelyonpatch_level_2_split_train_x.h5", "r")
    file_train_labels = h5py.File("train/camelyonpatch_level_2_split_train_y.h5", "r")
    file_test_pictures = h5py.File("test/camelyonpatch_level_2_split_test_x.h5", "r")
    file_test_labels = h5py.File("test/camelyonpatch_level_2_split_test_y.h5", "r")
    # Training
    dset_train_pictures = file_train_pictures["x"]
    dset_train_labels = file_train_labels["y"]
    # Test
    dset_test_pictures = file_test_pictures["x"]
    dset_test_labels = file_test_labels["y"]

    labels_training = []
    labels_testing = []
    for item in dset_train_labels:
        labels_training.append(item[0][0][0])

    for item in dset_test_labels:
        labels_testing.append(item[0][0][0])


    normalize_and_configure(dset_train_pictures, labels_training, dset_test_pictures, labels_testing)


def plot_roc_curve(y_true, y_score):
    # Assuming y_true are true labels and y_score are predicted probabilities
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    configure_dataset()