import mxnet as mx
import numpy as np

from test_utils.predict import predict


def isprs_pred(img, net, step, ctx):
    h, w, _ = img.shape
    predictions = np.zeros((h,w), dtype=np.uint8)
    img = np.transpose(img, axes=(2, 0, 1)).astype(np.float32)
    img /= 255
    img = mx.nd.array(np.expand_dims(img, 0), ctx=ctx)
    for row in range(0, h, step):
        for col in range(0, w, step):
            y = row if row + step <= h else h - step - 1
            x = col if col + step <= w else w - step - 1
            predictions[y:y+step, x:x+step] = predict(img[:, :, y:y+step, x:x+step], net)
    return predictions
