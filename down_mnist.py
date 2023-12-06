import urllib.request
import os
import os
import gzip
import numpy as np


def download_mnist():
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    data_dir = os.path.join(os.getcwd(), "mnist_data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for file in files:
        file_url = base_url + file
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(file_url, file_path)
            print("Download complete!")

download_mnist()


def load_mnist_data(data_dir):
    """Load MNIST data from the downloaded files."""
    with gzip.open(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), "rb") as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"), "rb") as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "rb") as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    return (x_train, y_train), (x_test, y_test)

# Replace 'mnist_data' with the path to your MNIST data directory
data_dir = os.path.join(os.getcwd(), "mnist_data")

# Load MNIST data
(x_train, y_train), (x_test, y_test) = load_mnist_data(data_dir)

# Print the shape of the data arrays to verify
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


