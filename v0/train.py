import os
import glob

import numpy as np
import tifffile as tif
import mxnet as mx

from mxnet import nd, autograd
from data_utils.data_loader import Isprs
from v0.model import Unet
from data_utils.k_fold import divide

root_path = '/media/xxx/Data/isprs/vaihingen'
datalist = os.listdir(os.path.join(root_path, 'gts_for_participants'))
k = 5
mask = [True] * 5
k_fold_idx = divide(list(range(len(datalist))), k=k)
val_idx = 0
mask[val_idx] = False
assert val_idx < k
val_idx = k_fold_idx[val_idx]
val_idx = val_idx.astype(np.int32)
k_fold_idx = k_fold_idx[mask]
train_idx = np.array([])
for fold in k_fold_idx:
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
train_data = mx.gluon.data.DataLoader(Isprs(images[train_idx], labels[train_idx], step, training=True),
                                      batch_size=min(24, len(train_idx)), shuffle=True,
                                      last_batch='rollover')
val_data = mx.gluon.data.DataLoader(Isprs(images[val_idx], labels[val_idx], step), batch_size=len(val_idx),
                                    last_batch='keep')
net = Unet(6)
ckpt_name = 'ckpt'
if os.path.exists(ckpt_name):
    net.load_params(ckpt_name, ctx=ctx, allow_missing=False)
    print('load params from ckpt')
else:
    net.collect_params().initialize(init=mx.init.Xavier(magnitude=2.24), ctx=ctx)

softmax_loss = mx.gluon.loss.SoftmaxCELoss(axis=1)
lr = 0.01
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': 0.0001})
print('learning rate {}'.format(lr))


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


epochs = 10000
smoothing_constant = .01

try:
    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_loss(output, label)
            loss.backward()
            trainer.step(data.shape[0])

            ##########################
            #  Keep a moving average of the losses
            ##########################
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
        if e % 100 == 0:
            test_accuracy = evaluate_accuracy(val_data, net)
            train_accuracy = evaluate_accuracy(train_data, net)
            print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
        else:
            train_accuracy = evaluate_accuracy(train_data, net)
            print("Epoch %s. Loss: %s, Train_acc %s" % (e, moving_loss, train_accuracy))
except KeyboardInterrupt:
    net.save_params(ckpt_name)
    print('params saved to ckpt')
except Exception as error:
    print('other error detected: {}'.format(error))
    raise
else:
    net.save_params(ckpt_name)
