import numpy as np

# https://github.com/scipy/scipy/blob/69a7752/scipy/special/_logsumexp.py#L130-L214
def logsumexp(a, axis = None, b = None, keepdims = False, return_sign = False):
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.0  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis = axis, keepdims = True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide = 'ignore'):
        s = np.sum(tmp, axis = axis, keepdims = keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis = axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def softmax(x, axis = None):
    return np.exp(x - logsumexp(x, axis = axis, keepdims = True))


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))
