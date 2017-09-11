from mxnet import gluon


class Fx(gluon.HybridBlock):
    def __init__(self, c, *args):
        super(Fx, self).__init__(*args)
        with self.name_scope():
            self.bn1 = gluon.nn.BatchNorm()
            self.act1 = gluon.nn.Activation(activation='relu')
            self.conv1 = gluon.nn.Conv2D(channels=c, kernel_size=3, padding=(1, 1))
            self.bn2 = gluon.nn.BatchNorm()
            self.act2 = gluon.nn.Activation(activation='relu')
            self.conv2 = gluon.nn.Conv2D(channels=c, kernel_size=3, padding=(1, 1))
            self.pool = gluon.nn.AvgPool2D(pool_size=(5, 5), strides=(1, 1), padding=(2, 2))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        px = x
        # px = self.pool(x)
        return x, px


class Down(gluon.HybridBlock):
    def __init__(self, c, *args):
        super(Down, self).__init__(*args)
        with self.name_scope():
            self.bn = gluon.nn.BatchNorm()
            self.act = gluon.nn.Activation('relu')
            self.conv = gluon.nn.Conv2D(channels=c, kernel_size=4, strides=(2, 2), padding=(1, 1))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.bn(x)
        x = self.act(x)
        return self.conv(x)


class Up(gluon.HybridBlock):
    def __init__(self, c, *args):
        super(Up, self).__init__(*args)
        with self.name_scope():
            self.bn = gluon.nn.BatchNorm()
            self.act = gluon.nn.Activation('relu')
            self.dconv = gluon.nn.Conv2DTranspose(channels=c, kernel_size=4, strides=(2, 2), padding=(1, 1))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.bn(x)
        x = self.act(x)
        return self.dconv(x)


class Unet(gluon.HybridBlock):
    def __init__(self, classes, *args):
        super(Unet, self).__init__(*args)
        with self.name_scope():
            self.start_conv = gluon.nn.Conv2D(channels=32, kernel_size=3, padding=(1, 1))
            self.f1 = Fx(32)
            self.d1 = Down(32)
            self.f2 = Fx(64)
            self.d2 = Down(64)
            self.f3 = Fx(128)
            self.d3 = Down(128)
            self.f4 = Fx(256)
            self.d4 = Down(256)
            self.f5 = Fx(512)
            self.u1 = Up(256)
            self.f6 = Fx(256)
            self.u2 = Up(128)
            self.f7 = Fx(128)
            self.u3 = Up(64)
            self.f8 = Fx(64)
            self.u4 = Up(32)
            self.f9 = Fx(32)
            self.out_bn = gluon.nn.BatchNorm()
            self.out_conv = gluon.nn.Conv2D(channels=classes, kernel_size=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.start_conv(x)

        x1, x = self.f1(x)
        x = self.d1(x)

        x2, x = self.f2(x)
        x = self.d2(x)

        x3, x = self.f3(x)
        x = self.d3(x)

        x4, x = self.f4(x)
        x = self.d4(x)

        x5, x = self.f5(x)

        x = self.u1(x)
        x = F.concat(x, x4, dim=1)
        _, x = self.f6(x)

        x = self.u2(x)
        x = F.concat(x, x3, dim=1)
        _, x = self.f7(x)

        x = self.u3(x)
        x = F.concat(x, x2, dim=1)
        _, x = self.f8(x)

        x = self.u4(x)
        x = F.concat(x, x1, dim=1)
        _, x = self.f9(x)

        x = self.out_bn(x)
        return self.out_conv(x)


if __name__ == '__main__':
    import mxnet as mx
    import numpy as np

    n = Unet(3)
    n.initialize(ctx=mx.gpu())
    # n.hybridize()
    sym = n(mx.nd.array(np.random.random((1, 3, 224, 224)), ctx=mx.gpu()))
    print(sym)
    # mx.viz.plot_network(symbol=sym, shape={'data': (1,3,224,224)}).view()
