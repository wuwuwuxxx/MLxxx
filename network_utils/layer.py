from mxnet import gluon


class Selu(gluon.HybridBlock):
    """
    selu activation from https://arxiv.org/abs/1706.02515
    Self-Normalizing Neural Networks
    """

    def __init__(self, **kwargs):
        super(Selu, self).__init__(**kwargs)
        with self.name_scope():
            self.alpha = 1.6732632423543772848170429916717
            self.scale = 1.0507009873554804934193349852946

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.LeakyReLU(data=x, act_type='elu', slope=self.alpha) * self.scale


class ConvBnAct(gluon.HybridBlock):
    def __init__(self, c, k=3, s=(1, 1), p=(1, 1), act=True, **kwargs):
        super(ConvBnAct, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = gluon.nn.Conv2D(c, k, strides=s, padding=p)
            self.bn = gluon.nn.BatchNorm()
            self.act = gluon.nn.LeakyReLU(0.25)
            self._act = act

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        if self._act:
            x = self.act(x)
        return x


class BnActConv(gluon.HybridBlock):
    def __init__(self, c, k=3, s=(1, 1), p=(1, 1), **kwargs):
        super(BnActConv, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = gluon.nn.Conv2D(c, k, strides=s, padding=p)
            self.bn = gluon.nn.BatchNorm()
            self.act = gluon.nn.LeakyReLU(0.25)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class ResBlock(gluon.HybridBlock):
    def __init__(self, c, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = ConvBnAct(c)
            self.conv2 = ConvBnAct(c, act=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x0 = x
        x = self.conv1(x)
        x = self.conv2(x)
        if x.shape == x0.shape:
            x += x0
        return F.LeakyReLU(data=x, slope=0.25, act_type='leaky')


class ResBlockBottleNeck(gluon.HybridBlock):
    def __init__(self, c, **kwargs):
        super(ResBlockBottleNeck, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = ConvBnAct(c, k=1, p=(0, 0))
            self.conv2 = ConvBnAct(c)
            self.conv3 = ConvBnAct(4 * c, k=1, p=(0, 0), act=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x0 = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if x.shape == x0.shape:
            x += x0
        return F.LeakyReLU(data=x, slope=0.25, act_type='leaky')


class ResBlocks(gluon.HybridBlock):
    def __init__(self, c, k, bottleneck=False, **kwargs):
        super(ResBlocks, self).__init__(**kwargs)
        if bottleneck:
            block = ResBlockBottleNeck
        else:
            block = ResBlock
        with self.name_scope():
            self.f = gluon.nn.HybridSequential()
            for _ in range(k):
                self.f.add(block(c))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.f(x)
        return x, x


if __name__ == '__main__':
    import mxnet as mx
    import numpy as np

    x = mx.nd.array(np.random.random((1, 3, 224, 224)), ctx=mx.cpu())
    f = Selu()
    print(f(x))

    f = ConvBnAct(32)
    f.initialize(ctx=mx.cpu())
    print(f(x))

    f = BnActConv(32)
    f.initialize(ctx=mx.cpu())
    print(f(x))

    f = ResBlock(32)
    f.initialize(ctx=mx.cpu())
    print(f(x))

    f = ResBlockBottleNeck(32)
    f.initialize(ctx=mx.cpu())
    print(f(x))

    f = ResBlocks(32, 3)
    f.initialize(ctx=mx.cpu())
    print(f(x))
