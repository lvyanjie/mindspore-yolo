#coding:utf-8

def autopad(k, p=None):
    """
    :param k: kernel
    :param p: padding
    """
    if p is None:
        p = k // 2 if isinstance(k, int) else(x // 2 for x in k)   # 默认是same
    return p
