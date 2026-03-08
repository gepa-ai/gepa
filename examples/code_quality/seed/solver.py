"""Stats utility module."""

import math


def c(d):
    s = 0
    for i in range(len(d)):
        s = s + d[i]
    return s / len(d)


def f(d, t=0):
    m = c(d)
    s = 0
    for i in range(len(d)):
        s = s + (d[i] - m) ** 2
    v = s / len(d)
    if t == 1:
        return v
    else:
        return math.sqrt(v)


def g(d):
    n = len(d)
    if n == 0:
        return None
    d2 = []
    for i in range(n):
        d2.append(d[i])
    for i in range(n):
        for j in range(n - 1):
            if d2[j] > d2[j + 1]:
                tmp = d2[j]
                d2[j] = d2[j + 1]
                d2[j + 1] = tmp
    if n % 2 == 0:
        return (d2[n // 2 - 1] + d2[n // 2]) / 2
    else:
        return d2[n // 2]


def h(d):
    if len(d) == 0:
        return None
    mx = d[0]
    mn = d[0]
    for i in range(1, len(d)):
        if d[i] > mx:
            mx = d[i]
        if d[i] < mn:
            mn = d[i]
    return mx - mn


def p(d, k):
    if len(d) == 0:
        return None
    d2 = []
    for i in range(len(d)):
        d2.append(d[i])
    for i in range(len(d2)):
        for j in range(len(d2) - 1):
            if d2[j] > d2[j + 1]:
                tmp = d2[j]
                d2[j] = d2[j + 1]
                d2[j + 1] = tmp
    idx = (k / 100) * (len(d2) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(d2):
        return d2[lo]
    w = idx - lo
    return d2[lo] * (1 - w) + d2[hi] * w


def z(d):
    m = c(d)
    s = f(d)
    r = []
    for i in range(len(d)):
        if s == 0:
            r.append(0)
        else:
            r.append((d[i] - m) / s)
    return r


def corr(d1, d2):
    if len(d1) != len(d2):
        return None
    n = len(d1)
    if n == 0:
        return None
    m1 = c(d1)
    m2 = c(d2)
    s1 = f(d1)
    s2 = f(d2)
    if s1 == 0 or s2 == 0:
        return None
    total = 0
    for i in range(n):
        total = total + (d1[i] - m1) * (d2[i] - m2)
    return total / (n * s1 * s2)


def report(d):
    result = {}
    result["mean"] = c(d)
    result["std"] = f(d)
    result["variance"] = f(d, 1)
    result["median"] = g(d)
    result["range"] = h(d)
    result["p25"] = p(d, 25)
    result["p75"] = p(d, 75)
    result["p90"] = p(d, 90)
    result["z_scores"] = z(d)
    result["iqr"] = result["p75"] - result["p25"]
    return result
