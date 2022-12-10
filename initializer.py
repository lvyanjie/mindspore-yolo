#coding:utf-8
from mindspore import nn
import mindspore as ms

import math


def default_recurisive_init(custom_cell):
    """Initialize parameter."""
    for _, cell in custom_cell.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Dense)):
            cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.HeUniform(math.sqrt(5)),
                                                                   cell.weight.shape, cell.weight.dtype))