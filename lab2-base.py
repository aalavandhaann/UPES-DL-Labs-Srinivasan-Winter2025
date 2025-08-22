import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from typing import Tuple

def load_and_preprocess_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape for CNN input: (batch, height, width, channels)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    

    return ( (x_train, y_train), (x_test, y_test) )

def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> models.Model:
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def compile_and_train_model(model: models.Model, 
                            x_train: np.ndarray, 
                            y_train: np.ndarray, 
                            x_test: np.ndarray, 
                            y_test: np.ndarray) -> models.Model:
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=64,
              validation_data=(x_test, y_test))
    return model

def evaluate_model(model: models.Model, x_test: np.ndarray, y_test: np.ndarray) -> None:
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=10)
    model = compile_and_train_model(model, x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)
