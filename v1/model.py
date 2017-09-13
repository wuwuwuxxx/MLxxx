from mxnet import gluon

from network_utils.layer import ResBlock


class Fx(gluon.HybridBlock):
    def __init__(self, c, k, *args):
        super(Fx, self).__init__(*args)
        with self.name_scope():
            self.f = gluon.nn.HybridSequential()
            for _ in range(k):
                self.f.add(ResBlock(c))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.f(x)
        return x, x


class Down(gluon.HybridBlock):
    def __init__(self, c, *args):
        super(Down, self).__init__(*args)
        with self.name_scope():
            self.bn = gluon.nn.BatchNorm()
            self.act = gluon.nn.LeakyReLU(0.25)
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
            self.act = gluon.nn.LeakyReLU(0.25)
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
            self.f1 = Fx(32, 2)
            self.d1 = Down(64)
            self.f2 = Fx(64, 3)
            self.d2 = Down(128)
            self.f3 = Fx(128, 4)
            self.d3 = Down(256)
            self.f4 = Fx(256, 5)
            self.d4 = Down(512)
            self.f5 = Fx(512, 6)
            self.u1 = Up(256)
            self.f6 = Fx(256, 5)
            self.u2 = Up(128)
            self.f7 = Fx(128, 4)
            self.u3 = Up(64)
            self.f8 = Fx(64, 3)
            self.u4 = Up(32)
            self.f9 = Fx(32, 2)
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

    x = mx.nd.array(np.random.random((1, 3, 224, 224)), ctx=mx.cpu())
    f = Unet(6)
    f.initialize(ctx=mx.cpu())
    print(f(x))
