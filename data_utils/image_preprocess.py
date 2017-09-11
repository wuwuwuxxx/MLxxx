import cv2
import numpy as np


def color_jit(img, h=0.05, s=0.05, v=0.1):
    """
    :param img: a RGB image, type np.uint8
    :param h: random number add to h, from -h to h, hue
    :param s: random number add to s, from -s to s, saturation
    :param v: random number add to v, from -v to v, value
    :return:
    """
    # color jit
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv /= 255
    hsv[:, :, 0] += np.random.uniform(-h, h)
    hsv[:, :, 1] += np.random.uniform(-s, s)
    hsv[:, :, 2] += np.random.uniform(-v, v)
    hsv = np.clip(hsv, 0, 1)
    hsv *= 255
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
