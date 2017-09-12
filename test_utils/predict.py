import mxnet as mx
import numpy as np


def predict(img, net):
    predictions = mx.nd.argmax(net(img), axis=1).asnumpy()
    predictions = np.squeeze(predictions)
    return predictions.astype(np.int64)
