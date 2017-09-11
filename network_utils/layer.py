from mxnet import gluon


class Selu(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Selu, self).__init__(**kwargs)
        with self.name_scope():
            self.alpha = 1.6732632423543772848170429916717
            self.scale = 1.0507009873554804934193349852946

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.LeakyReLU(data=x, act_type='elu', slope=self.alpha) * self.scale
