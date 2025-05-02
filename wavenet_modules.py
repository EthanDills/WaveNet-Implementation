import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
import numpy as np


def dilate(x, dilation, init_dilation=1, pad_start=True):
    print("dilate() called")
    """
    :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    [n, c, l] = x.size()
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l_old = int(round(l / dilation_factor))
    n_old = int(round(n * dilation_factor))
    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        #       |
        #  |6|7|8|1|2|3|4|5|
        #         |
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0


# class ConstantPad1d(Function): 

    # def __init__(self, target_size, dimension=0, value=0, pad_start=False):
    #     print("no error yet!")
    #     super(ConstantPad1d, self).__init__()
    #     self.target_size = target_size
    #     self.dimension = dimension
    #     self.value = value
    #     self.pad_start = pad_start

    # def forward(self, input): # this is giving me the autograd Attribute Error
    #     self.num_pad = self.target_size - input.size(self.dimension)
    #     assert self.num_pad >= 0, 'target size has to be greater than input size'

    #     self.input_size = input.size()

    #     size = list(input.size())
    #     size[self.dimension] = self.target_size
    #     output = input.new(*tuple(size)).fill_(self.value)
    #     c_output = output

    #     # crop output
    #     if self.pad_start:
    #         c_output = c_output.narrow(self.dimension, self.num_pad, c_output.size(self.dimension) - self.num_pad)
    #     else:
    #         c_output = c_output.narrow(self.dimension, 0, c_output.size(self.dimension) - self.num_pad)

    #     c_output.copy_(input)
    #     return output

    # def backward(self, grad_output): # this will ALSO give me the autograd Attribute Error
    #     grad_input = grad_output.new(*self.input_size).zero_()
    #     cg_output = grad_output

    #     # crop grad_output
    #     if self.pad_start:
    #         cg_output = cg_output.narrow(self.dimension, self.num_pad, cg_output.size(self.dimension) - self.num_pad)
    #     else:
    #         cg_output = cg_output.narrow(self.dimension, 0, cg_output.size(self.dimension) - self.num_pad)

    #     grad_input.copy_(cg_output)
    #     return grad_input


class ConstantPad1d(Function):
    @staticmethod
    def forward(ctx, input, target_size, dimension=0, value=0, pad_start=False):
        ctx.input_size = input.size()
        ctx.dimension = dimension
        ctx.pad_start = pad_start
        ctx.target_size = target_size

        num_pad = target_size - input.size(dimension)
        assert num_pad >= 0, "target size must be >= input size"

        # Create padded tensor
        size = list(input.size())
        size[dimension] = target_size
        output = input.new_full(size, value)

        if pad_start:
            output.narrow(dimension, num_pad, input.size(dimension)).copy_(input)
        else:
            output.narrow(dimension, 0, input.size(dimension)).copy_(input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_zeros(ctx.input_size)
        if ctx.pad_start:
            grad_output = grad_output.narrow(ctx.dimension, ctx.target_size - ctx.input_size[ctx.dimension], ctx.input_size[ctx.dimension])
        else:
            grad_output = grad_output.narrow(ctx.dimension, 0, ctx.input_size[ctx.dimension])

        grad_input.copy_(grad_output)
        return grad_input, None, None, None, None  # one gradient per forward argument


def constant_pad_1d(input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False):
    return ConstantPad1d.apply(input, target_size, dimension, value, pad_start)
