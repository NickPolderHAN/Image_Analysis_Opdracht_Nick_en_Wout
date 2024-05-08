import matplotlib
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import h5py
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def train_model(train_data, test_data):
    batch_size = 150
    img_height = 96
    img_width = 96

    tf.config.list_devices(device_type="GPU")

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
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()

    epochs = 5
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


def format_dataset(train_ds, labels_training, test_ds, labels_testing):
    # Assuming train_ds and test_ds are NumPy arrays
    train_ds_ = tf.data.Dataset.from_tensor_slices((train_ds, labels_training))
    test_ds_ = tf.data.Dataset.from_tensor_slices((test_ds, labels_testing))

    return train_ds_, test_ds_


def configure_dataset():
    # x zijn de plaatjes, y zijn de binaire labels (wel of niet)
    file_train_pictures = h5py.File("train/camelyonpatch_level_2_split_train_x.h5", "r")
    file_train_labels = h5py.File("train/camelyonpatch_level_2_split_train_y.h5", "r")
    file_test_pictures = h5py.File("test/camelyonpatch_level_2_split_test_x.h5", "r")
    file_test_labels = h5py.File("test/camelyonpatch_level_2_split_test_y.h5", "r")

    # Training dataframes.
    dset_train_pictures = file_train_pictures["x"]
    dset_train_labels = file_train_labels["y"]

    # Test dataframes.
    dset_test_pictures = file_test_pictures["x"]
    dset_test_labels = file_test_labels["y"]

    # Store the different training and testing labels.
    labels_training = []
    labels_testing = []

    # Gets the different items from the training set labels.
    for item in dset_train_labels:
        labels_training.append(item[0][0][0])

    # Gets the different items from the testing set labels.
    for item in dset_test_labels:
        labels_testing.append(item[0][0][0])

    return dset_train_pictures, labels_training, dset_test_pictures, labels_testing


def plot_roc_curve(y_true, y_score):
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
    with tf.device("/GPU:0"):
        dset_train_pictures, labels_training, dset_test_pictures, labels_testing = configure_dataset()
        train_ds, test_ds = format_dataset(dset_train_pictures, labels_training,
                                                dset_test_pictures, labels_testing)
        train_model(train_ds, test_ds)
