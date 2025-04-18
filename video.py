import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input



EPOCHS = 30
IMG_WIDTH = 30
IMG_HEIGHT = 30

#can use this if the categories is numbered starting from 0...
#NUM_CATEGORIES = 2
TEST_SIZE = 0.2


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    datagen = ImageDataGenerator(
        shear_range=0.05,  # Reduced shear
        zoom_range=0.1,    # Reduced zoom
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Slightly reduced brightness variation
        channel_shift_range=0.05,     # Reduced channel shift
        fill_mode="nearest"
    )

    # Fit data augmentation only on training set
    #datagen.fit(x_train)

    # Get a compiled neural network
    model = get_model()
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        patience=5,          # Stop if no improvement after 5 epochs
        restore_best_weights=True,  # Restore the best weights
        verbose=1
    )
    # Fit model on training data
    model.fit(x_train, y_train, 
        batch_size=32, epochs=EPOCHS, 
        validation_data=(x_test, y_test),  # Pass the learning rate scheduler here
        callbacks=[lr_scheduler, early_stopping]  # Use both callbacks

    )

    print(model.summary())
    # Evaluate neural network performance
    print(model.evaluate(x_test,  y_test, verbose=2))

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    categories = os.listdir(data_dir)  # Getting actual category names

    # Iterate over category directories
    for category in categories:
        category_path = os.path.join(data_dir, str(category))  # Path to category folder
        
        # Check if category directory exists
        if not os.path.isdir(category_path):
            continue  # Skip missing categories

        # Iterate over image files in the category directory
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)

            # Read the image using OpenCV
            image = cv2.imread(file_path)
            if image is None:
                print(f"Warning: Unable to read file {file_path}. Skipping.")

                continue  # Skip unreadable files

            # Resize the image to standard dimensions
            imageL = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = imageL / 255.0  # Normalize to [0, 1]

            # Append to lists
            images.append(image)
            labels.append(category)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = Sequential([
        Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),  # Explicit Input layer

        # Convolutional layer 1
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Convolutional layer 2
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Convolutional layer 3
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Flatten
        Flatten(),

        # Fully connected layer
        Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.5),

        # Fully connected layer
        Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.4),

        # Output layer
        Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
