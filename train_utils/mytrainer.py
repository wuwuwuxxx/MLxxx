import mxnet as mx

from mxnet import nd, autograd


class MyTrainer(mx.gluon.Trainer):
    def __init__(self, net=None, train_data_iter=None, val_data_iter=None, loss=None, ckpt_name='ckpt',
                 ctx=mx.gpu(0), decay_epochs=10, do_ckpt_epochs=1, train_epochs=10000, **kwargs):
        super(MyTrainer, self).__init__(**kwargs)
        self._net = net
        self._train_data_iter = train_data_iter
        self._val_data_iter = val_data_iter
        self._val_epoch = 0
        self._ctx = ctx
        self._best_acc = 0
        self._decay_epochs = decay_epochs
        self._ckpt_name = ckpt_name
        self._loss = loss
        self._train_epochs = train_epochs
        self._do_ckpt_epochs = do_ckpt_epochs

    def evaluate_accuracy(self, data_iter):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self._ctx)
            label = label.as_in_context(self._ctx)
            output = self._net(data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        self._val_epoch += 1
        val_acc = acc.get()[1]
        if val_acc > self._best_acc:
            self._val_epoch = 0
            self._best_acc = val_acc
            self._net.save_params(self._ckpt_name)
        if self._val_epoch == self._decay_epochs:
            self.set_learning_rate(self.learning_rate / 2)
            print('set learning rate to {}'.format(self.learning_rate))
            # self._net.load_params(self._ckpt_name)
        return val_acc

    def train(self):
        smoothing_constant = 0.01
        e = 0
        moving_loss = None
        while True:
            for i, (data, label) in enumerate(self._train_data_iter):
                data = data.as_in_context(self._ctx)
                label = label.as_in_context(self._ctx)
                with autograd.record():
                    output = self._net(data)
                    loss = self._loss(output, label)
                loss.backward()
                self.step(data.shape[0])

                ##########################
                #  Keep a moving average of the losses
                ##########################
                curr_loss = nd.mean(loss).asscalar()
                moving_loss = (curr_loss if moving_loss is None else (1 - smoothing_constant) * moving_loss +
                                                                     (smoothing_constant) * curr_loss)
            if e % self._do_ckpt_epochs == 0:
                test_accuracy = self.evaluate_accuracy(self._val_data_iter)
                print("Epoch {}. Loss: {}, Test_acc {}, Learning rate: {}".format(
                    e, moving_loss, test_accuracy, self.learning_rate))
            else:
                print("Epoch %s. Loss: %s" % (e, moving_loss))
            e += 1
            if self.learning_rate < 1e-7:
                break
        print('best validation accuracy is {}'.format(self._best_acc))
