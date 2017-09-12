import os
import glob

import numpy as np
import tifffile as tif
import mxnet as mx

from data_utils.data_loader import Isprs
from data_utils.k_fold import divide
from test_utils.metrics import accuracy
##################################################################################
from v0.model import Unet# change v0 here for different model
##################################################################################

root_path = '/media/xxx/Data/isprs/vaihingen'
datalist = os.listdir(os.path.join(root_path, 'gts_for_participants'))
k = 5
mask = [True] * k
k_fold_idx = divide(list(range(len(datalist))), k=k)

val_idx = 0
mask[val_idx] = False
assert val_idx < k
val_idx = k_fold_idx[val_idx]
val_idx = val_idx.astype(np.int32)
temp = k_fold_idx[mask]
train_idx = np.array([])
for fold in temp:
    train_idx = np.append(train_idx, fold)
train_idx = train_idx.astype(np.int32)
# read images
if not os.path.exists('../images.npy'):
    paths = [os.path.join(root_path, 'top', name) for name in datalist]
    images = [tif.imread(p) for p in paths]
    np.save('../images.npy', images)
else:
    images = np.load('../images.npy')
# read labels
if not os.path.exists('../labels.npy'):
    paths = glob.glob(os.path.join(root_path, 'gts_for_participants', '*.tif'))
    labels = [tif.imread(p) for p in paths]
    l = list()
    for label in labels:
        h, w, _ = label.shape
        t = np.zeros((h, w), dtype=np.uint8)
        t[np.all(label == [0, 0, 255], axis=-1)] = 1
        t[np.all(label == [0, 255, 255], axis=-1)] = 2
        t[np.all(label == [0, 255, 0], axis=-1)] = 3
        t[np.all(label == [255, 255, 0], axis=-1)] = 4
        t[np.all(label == [255, 0, 0], axis=-1)] = 5
        l.append(t)
    np.save('../labels.npy', l)
else:
    labels = np.load('../labels.npy')

ctx = mx.gpu()
step = 32 * 8
val_data = mx.gluon.data.DataLoader(Isprs(images[val_idx], labels[val_idx], step), batch_size=len(val_idx),
                                        last_batch='keep')
net = Unet(6)
ckpt_name = 'ckpt_{}'.format(0)
if os.path.exists(ckpt_name):
    net.load_params(ckpt_name, ctx=ctx, allow_missing=False)
    print('load params from ckpt')
else:
    net.collect_params().initialize(init=mx.init.MSRAPrelu('in'), ctx=ctx)
net.hybridize()
print(accuracy(val_data, ctx, net))