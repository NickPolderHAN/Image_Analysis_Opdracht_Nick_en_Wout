import keras
from keras import layers, Sequential, Input
from medmnist import ChestMNIST
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def get_dataset():
    """
    Gathers the training dataset and splits the test and validation subsets from it.

    :return:
    - pictures_train -> Contains the pictures used for training.
    - labels_train -> Contains the labels used for training.
    - pictures_valid -> Contains the pictures used for validation.
    - labels_valid -> Contains the labels used for validation.
    - pictures_test -> Contains the pictures used for testing.
    - labels_test -> Contains the labels used for testing.
    """
    # Lists to store pictures and labels
    pictures = []
    labels = []
    # Data ophalen en splitsen in train en test sets.
    dataset_train = ChestMNIST(split='train', download=True, size=128)

    # Populate pictures and labels for training
    for image, label in dataset_train:
        pictures.append(np.array(image))
        labels.append(1 if 1 in label else 0)
    # Split the data into training, validation, and test sets.
    pictures_train, pictures_remaining, labels_train, labels_remaining = train_test_split(pictures, labels,
                                                                                          test_size=0.2,
                                                                                          random_state=42)
    # Split the remaining data into validation and test sets.
    pictures_valid, pictures_test, labels_valid, labels_test = train_test_split(pictures_remaining, labels_remaining,
                                                                                test_size=0.5, random_state=42)
    return pictures_train, labels_train, pictures_valid, labels_valid, pictures_test, labels_test


def merge_picture_and_label(pictures_train, labels_train, pictures_valid,
                            labels_valid, pictures_test, labels_test):
    """
    Merges the labels and pictures of the training, testing and validation sets together
    into their own respective sets.

    :param pictures_train: Contains the training pictures.
    :param labels_train: Contains the labels for training.
    :param pictures_valid: Contains the pictures for validation.
    :param labels_valid: Contains the labels for validation.
    :param pictures_test: Contains the pictures for testing.
    :param labels_test: Contains the labels for testing.
    :return:
    - train_ds_ -> Contains the training dataset
    - test_ds_ -> Contains the test dataset.
    - val_ds_ -> Contains the validation dataset.
    """
    batch_size = 100

    # Convert to Tensorflow dataset.
    train_ds_ = tf.data.Dataset.from_tensor_slices(
        (np.array(pictures_train), np.array(labels_train)))  # Convert to numpy array

    # Convert to Tensorflow dataset.
    test_ds_ = tf.data.Dataset.from_tensor_slices(
        (np.array(pictures_test), np.array(labels_test)))  # Convert to numpy array

    # Convert to Tensorflow dataset.
    val_ds_ = tf.data.Dataset.from_tensor_slices((np.array(pictures_valid), np.array(labels_valid)))

    # Set batch size for each data sample.
    train_ds_ = train_ds_.batch(batch_size)
    test_ds_ = test_ds_.batch(batch_size)
    val_ds_ = val_ds_.batch(batch_size)

    return train_ds_, test_ds_, val_ds_


def model_training(train_data, test_data):
    """
    Trains a Convolutional Neural Network (CNN) model on the provided training dataset
    and evaluates it on the test dataset.

    :param train_data: tf.data.Dataset
        The training dataset containing image data and corresponding labels.

    :param test_data: tf.data.Dataset
        The test dataset containing image data and corresponding labels.

    :return: None
    """
    # Store image sizes.
    img_height = 128
    img_width = 128

    # Creating data augmentation layers
    data_augmentation = Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),  # Reduce rotation degree for medical images
            layers.RandomZoom(0.2),  # Include random zoom
            layers.RandomContrast(0.2),  # Adjust contrast
            layers.RandomBrightness(0.2),  # Adjust brightness
            layers.RandomTranslation(0.1, 0.1),  # Translate images slightly
        ]
    )

    # Building the model
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.BatchNormalization(axis=1, name='bn_conv1'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(124, 3, padding='same', activation='relu'),
        layers.Conv2D(124, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(124, activation='relu'),
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

    # Makes predictions using the test data.
    predictions = model.predict(test_data)

    # Assuming labels are binary (0 or 1)
    y_score = predictions.flatten()
    y_true = test_data.map(lambda x, y: y).unbatch()  # Extract true labels
    y_true = [label.numpy() for label in y_true]

    # plot_roc_curve(y_true, y_score)

    # Store the accuracy history.
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Store the loss history.
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Gets the epoch range used.
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plot_roc_curve(y_true, y_score):
    """
    Plots the ROC curve of a trained model using the true labels and predicted scores.

    :param y_true: array-like of shape (n_samples,)
        True binary labels. Values should be either 0 or 1.

    :param y_score: array-like of shape (n_samples,)
        Target scores, which can either be probability estimates of the positive class,
        confidence values, or binary decisions.

    :return: None
    """
    # With the y_true being the
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curves
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
    pictures_train, labels_train, pictures_valid, labels_valid, pictures_test, labels_test = get_dataset()
    train_data, test_data, val_ds = merge_picture_and_label(pictures_train,
                                                            labels_train,
                                                            pictures_valid,
                                                            labels_valid,
                                                            pictures_test,
                                                            labels_test)
    model_training(train_data, test_data)
