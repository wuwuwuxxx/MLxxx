import numpy as np

from mxnet import gluon
from data_utils.divide_img import divide_img
from data_utils.image_preprocess import color_jit


class Isprs(gluon.data.Dataset):
    def __init__(self, inputs, labels, cropsize, training=False, data_augumentation=False):
        """
        :param inputs: a list of numpy array
        :param labels: a list of numpy array
        :param cropsize: cropped image size, type int
        :param training:
        :param data_augumentation:
        """
        assert len(inputs) == len(labels)
        if training:
            self._data = inputs
            self._label = labels
        else:
            self._data = divide_img(inputs, cropsize)
            self._label = divide_img(labels, cropsize)
        self._step = cropsize
        self._training = training
        self._data_augumentation = data_augumentation

    def __getitem__(self, item):
        d = self._data[item]
        l = self._label[item]
        if self._training:
            # crop
            h, w, _ = d.shape
            assert self._step < h
            assert self._step < w
            r = np.random.randint(0, h - self._step - 1, dtype=np.int32)
            c = np.random.randint(0, w - self._step - 1, dtype=np.int32)
            d = d[r:r + self._step, c:c + self._step, :]
            l = l[r:r + self._step, c:c + self._step]
            if self._data_augumentation:
                # flip
                if np.random.random() > 0.5:
                    d = np.fliplr(d)
                    l = np.fliplr(l)
                if np.random.random() > 0.5:
                    d = np.flipud(d)
                    l = np.flipud(l)
                    # # color jit
                    # d = color_jit(d)
        # normalization
        d = d.astype(np.float32)
        d /= 255.0
        # transpose
        d = np.transpose(d, axes=(2, 0, 1))
        return d, l

    def __len__(self):
        return len(self._data)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    c = Isprs(np.load('images.npy'), np.load('labels.npy'), 224, training=False)
    for k in range(10):
        d, l = c.__getitem__(k)
        d = np.transpose(d, axes=(1, 2, 0))
        f, axs = plt.subplots(1, 2)
        axs[0].imshow(d)
        axs[1].imshow(l)
        plt.show()
