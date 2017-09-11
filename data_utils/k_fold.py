"""
this script is used to divide the available data into
training data and validation data use k fold strategy.
"""
import numpy as np

np.random.seed(1)


def divide(permutation, k=5, labellist=None):
    """
    :param permutation: oneD array
    :param k:
    :param labellist: oneD array, index from 0
    :return:
    """
    if type(permutation) == list:
        permutation = np.array(permutation)
    assert permutation.shape[0] >= k
    if labellist is None:
        # balanced data
        length = len(permutation)
        np.random.shuffle(permutation)
        result = list()
        step = round(length / k)
        idx = 0
        for kk, v in enumerate(range(k)):
            temp = permutation[idx:idx + step] if kk != k - 1 else permutation[idx:]
            result.append(temp)
            idx += step
        return np.array(result)
    else:
        if type(labellist) == list:
            labellist = np.array(labellist)
        assert permutation.shape[0] == labellist.shape[0]
        # unbalanced data
        classes = np.max(labellist) + 1

        for kk, v in enumerate(range(classes)):
            p = permutation[labellist == v]
            if kk == 0:
                result = divide(p, k=k)
            else:
                temp = divide(p, k=k)
                for kk, v in enumerate(result):
                    result[kk] = np.append(v, temp[kk], axis=0)
        return result


if __name__ == '__main__':
    r = divide(list(range(19)))
    print(r)
    l = list(range(102))
    ll = np.random.randint(0, 5, size=(102,), dtype=np.int32)
    print(ll)
    fold = 6
    tt = divide(l, k=fold, labellist=ll)
    print(tt)
    for k in range(fold):
        print(ll[tt[k]])
