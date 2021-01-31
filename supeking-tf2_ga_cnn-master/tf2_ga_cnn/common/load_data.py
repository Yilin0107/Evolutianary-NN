import tensorflow.keras.datasets.mnist as mnist
import tensorflow.keras.datasets.cifar10 as cifar
import tensorflow as tf
import numpy as np
from common.utils import rgb2gray


from sklearn.datasets import load_iris
from sklearn.datasets import load_digits


# cifar
(cifar_train_x, cifar_train_y), (cifar_test_x, cifar_test_y) = cifar.load_data()
#cifar_train_x, cifar_test_x = rgb2gray(cifar_train_x), rgb2gray(cifar_test_x)
cifar_train_x, cifar_test_x = (cifar_train_x / 255.0).astype(np.float32), (cifar_test_x / 255.0).astype(np.float32)
# mnsit
(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data()
mnist_train_x, mnist_test_x = (mnist_train_x / 255.0).astype(np.float32), (mnist_test_x / 255.0).astype(np.float32)

#Iris
data = load_iris()
iris_data = np.float32(data.data)
iris_target = (data.target)
#iris_target = np.float32(tf.keras.utils.to_categorical(iris_target,num_classes=3))

#load_digits
#images: (1797, 8, 8)
digits_images, digits_targets = load_digits().images, load_digits().target

def digitsDataset(batch_size=1797, train=True):
    train_x, test_x = digits_images, digits_targets
    for i in range(train_x.shape[0]):
        if (i + batch_size) <= train_x.shape[0]:
            batch_data = train_x[i:i + batch_size, ...]
            batch_label = test_x[i:i + batch_size, ...]
        else:
            batch_data = train_x[i:, ...]
            batch_label = test_x[i:, ...]
        yield batch_data, batch_label


def mnsitDataset(batch_size=10000, train=True):
    train_x, test_x = mnist_train_x, mnist_test_x
    train_y, test_y = mnist_train_y, mnist_test_y
    if train:
        image, label = train_x, train_y
    else:
        image, label = test_x, test_y
    for i in range(0, image.shape[0], batch_size):
        if (i + batch_size)<=image.shape[0]:
            batch_image = image[i:i + batch_size, ...]
            batch_label = label[i:i + batch_size, ...]
        else:
            batch_image = image[i:, ...]
            batch_label = label[i:, ...]
        yield batch_image, batch_label


def cifarDataset(batch_size=10000, train=True):
    train_x, test_x = cifar_train_x, cifar_test_x
    train_y, test_y = cifar_train_y, cifar_test_y
    if train:
        image, label = train_x, train_y
    else:
        image, label = test_x, test_y
    for i in range(0, image.shape[0], batch_size):
        if (i + batch_size) <= image.shape[0]:
            batch_image = image[i:i + batch_size, ...]
            batch_label = label[i:i + batch_size, ...]
        else:
            batch_image = image[i:, ...]
            batch_label = label[i:, ...]
        yield batch_image, batch_label



def irisDataset(batch_size=150, train=True):
    train_x, test_x = iris_data, iris_target
    for i in range(train_x.shape[0]):
        if (i + batch_size) <= train_x.shape[0]:
            batch_data = train_x[i:i + batch_size, ...]
            batch_label = test_x[i:i + batch_size, ...]
        else:
            batch_data = train_x[i:, ...]
            batch_label = test_x[i:, ...]
        yield batch_data, batch_label


if __name__ == '__main__':

    ds = cifarDataset(batch_size=10000, train=True)
    x1, y1 = next(ds)
    print(x1.shape)
    print(y1.shape)
    print('-')

