import mxnet as mx


def accuracy(data_iter, ctx, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iter):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = mx.nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]