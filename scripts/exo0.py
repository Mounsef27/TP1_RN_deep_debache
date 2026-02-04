# exo0.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import utils as np_utils
from tensorflow.keras.datasets import mnist
def main():
    # Load MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten + normalize as in the TP
    X_train = X_train.reshape(60000, 784).astype("float32") / 255.0
    X_test  = X_test.reshape(10000, 784).astype("float32") / 255.0

    print(X_train.shape[0], "train samples")
    print(X_test.shape[0], "test samples")

    # Stats for the report
    print("X_train min:", X_train.min(), "max:", X_train.max())
    print("Input space: [0,1]^784 subset of R^784")

    # Show first 200 images
    plt.figure(figsize=(7.195, 3.841), dpi=100)
    for i in range(200):
        plt.subplot(10, 20, i + 1)
        plt.imshow(X_train[i, :].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.tight_layout(pad=0.1)
    plt.show()

if __name__ == "__main__":
    main()
