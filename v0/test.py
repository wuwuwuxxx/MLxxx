import os

import numpy as np
import tifffile as tif
import mxnet as mx

from test_utils.splice_image import isprs_pred
##################################################################################
from v0.model import Unet  # change v0 here for different model
##################################################################################

root_path = '/media/xxx/Data/isprs/vaihingen'
datalist = os.listdir(os.path.join(root_path, 'gts_for_participants'))

ctx = mx.gpu()
step = 32 * 8

net = Unet(6)
ckpt_name = 'ckpt_{}'.format(0)
if os.path.exists(ckpt_name):
    net.load_params(ckpt_name, ctx=ctx, allow_missing=False)
    print('load params from ckpt')
else:
    net.collect_params().initialize(init=mx.init.MSRAPrelu('in'), ctx=ctx)
net.hybridize()
output_dir = 'results'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
for img in os.listdir(os.path.join(root_path, 'top')):
    if img not in datalist:
        p = os.path.join(root_path, 'top', img)
        img = tif.imread(p)
        s = np.zeros_like(img)
        t = isprs_pred(img, net, step, ctx)
        s[t==0, :] = [255,255,255]
        s[t == 1, :] = [0, 0, 255]
        s[t == 2, :] = [0, 255, 255]
        s[t == 3, :] = [0, 255, 0]
        s[t == 4, :] = [255, 255, 0]
        s[t == 5, :] = [255, 0, 0]
        tif.imsave(os.path.join(output_dir, img), s)
