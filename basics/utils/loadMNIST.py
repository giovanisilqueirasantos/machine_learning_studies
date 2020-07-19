import os
import struct
import numpy as np

def load_mnist(kind='train'):
    """Load MNIST data and return two arrays with data and labels"""

    lables_path = os.path.join('datasets/mnist/', f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join('datasets/mnist/', f'{kind}-images-idx3-ubyte')

    with open(lables_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels